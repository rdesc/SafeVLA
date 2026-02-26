from typing import Any, Dict, List, Sequence, Optional, Set, Tuple

import gym
import numpy as np
from PIL import Image
from allenact.base_abstractions.sensor import Sensor

from utils.local_logging import unnormalize_image, WandbLoggingSensor

import wandb
from allenact.base_abstractions.callbacks import Callback
import os


class SimpleWandbLogging(Callback):
    def __init__(
        self,
        project: str,
        entity: str,
        log_videos_every_n_logs: int = 0,
        max_videos_per_log: int = 16,
        log_videos_from_device: Optional[int] = None,
    ):
        self.project = project
        self.entity = entity
        self.log_videos_every_n_logs = log_videos_every_n_logs
        self.max_videos_per_log = max_videos_per_log
        self.log_videos_from_device = log_videos_from_device

        self._defined_metrics: Set[str] = set()
        self._log_call_count: int = 0

    def setup(self, name: str, **kwargs) -> None:
        if "OUTPUT_DIR" in os.environ.keys() and "EXTRA_TAG" in os.environ.keys():
            possible_wandb_id = "{}/used_configs/{}.txt".format(
                os.environ["OUTPUT_DIR"], os.environ["EXTRA_TAG"]
            )
            if os.path.isfile(possible_wandb_id):
                with open(possible_wandb_id, "r") as f:
                    wandb_id = f.read()
                wandb_id = wandb_id.split("\n")[0]
            else:
                wandb_id = wandb.util.generate_id()
                with open(possible_wandb_id, "w") as f:
                    f.write(wandb_id)
            wandb.init(
                project=self.project,
                entity=self.entity,
                name=name,
                config=kwargs,
                id=wandb_id,
                resume="allow",
            )
        else:
            wandb.init(
                project=self.project,
                entity=self.entity,
                name=name,
                config=kwargs,
            )
        
        # Pre-define custom metrics
        wandb.define_metric("train-misc/house_index_device_0", step_metric="training_step")
        self._defined_metrics.add("train-misc/house_index_device_0")

    def _define_missing_metrics(
        self,
        metric_means: Dict[str, float],
        scalar_name_to_total_experiences_key: Dict[str, str],
    ):
        for k, v in metric_means.items():
            if k not in self._defined_metrics:
                wandb.define_metric(
                    k,
                    step_metric=scalar_name_to_total_experiences_key.get(k, "training_step"),
                )

                self._defined_metrics.add(k)

    def on_train_log(
        self,
        metrics: List[Dict[str, Any]],
        metric_means: Dict[str, float],
        step: int,
        tasks_data: List[Any],
        scalar_name_to_total_experiences_key: Dict[str, str],
        **kwargs,
    ) -> None:
        """Log the train metrics to wandb."""

        self._define_missing_metrics(
            metric_means=metric_means,
            scalar_name_to_total_experiences_key=scalar_name_to_total_experiences_key,
        )

        log_data = {
            **metric_means,
            "training_step": step,
        }

        # Log house_index from sampler_index 0 running on sampler_device 0.
        if metrics is not None:
            for metric in metrics:
                if not isinstance(metric, dict) or "task_info" not in metric:
                    continue
                task_info = metric["task_info"]
                sampler_device = int(task_info.get("sampler_device", -1))
                sampler_index = int(task_info.get("sampler_index", -1))

                if sampler_device == 0 and sampler_index == 0:
                    house_index = task_info.get("house_index")
                    if house_index is not None:
                        log_data["train-misc/house_index_device_0"] = float(house_index)
                        print(
                            f"[WANDB] house_index_device_0={house_index} "
                            f"(sampler_index={sampler_index}, sampler_device={sampler_device}) "
                            f"step={step}"
                        )
                    break

        wandb.log(log_data)

        self._log_call_count += 1
        print(f"[WANDB] {step}, log_call_count {self._log_call_count}, {self.log_videos_every_n_logs}")

        if (
            self.log_videos_every_n_logs > 0
            and self._log_call_count % self.log_videos_every_n_logs == 0
        ):
            print(f"[WANDB] Logging videos at step {step} (log call count: {self._log_call_count})")
            metrics_list, tasks_list = self._filter_tasks_for_video_logging(
                metrics=metrics, tasks_data=tasks_data
            )
            if len(metrics_list) > 0:
                (
                    table_content,
                    frames_with_logits_list,
                    videos_without_logit_list,
                ) = self.get_table_content(
                    metrics=metrics_list,
                    tasks_data=tasks_list,
                    frames_with_logit_flag=False,
                )

                video_dict = {"all_videos": {}}
                for vid, data in zip(frames_with_logits_list, table_content):
                    idx = data[-1]
                    video_dict["all_videos"][idx] = vid

                table = wandb.Table(
                    columns=[
                        "Trajectory",
                        "Path",
                        "Episode Length",
                        "Success",
                        "Dist to target",
                        "Task Type",
                        "House Name",
                        "Target Object Type",
                        "Prompt",
                        "Task Id",
                        "Index",
                    ]
                )

                for data in table_content:
                    table.add_data(*data)

                wandb.log(
                    {
                        "training_step": step,
                        "Train Qualitative Examples": table,
                        "Train Videos": video_dict,
                    }
                )

    def _extract_rgb_frame(self, obs: Dict[str, Any]) -> Optional[np.ndarray]:
        if "rgb_raw" in obs:
            frame = obs["rgb_raw"]
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            return frame
        if "rgb" in obs:
            frame = obs["rgb"]
            if frame.dtype == np.uint8:
                return frame
            frame = unnormalize_image(frame)
            return (frame * 255).astype(np.uint8)
        return None

    def combine_rgb_across_episode(self, observation_list):
        all_rgb = []
        for obs in observation_list:
            frame = self._extract_rgb_frame(obs)
            if frame is None:
                continue
            all_rgb.append(np.array(Image.fromarray(frame)))

        return all_rgb

    def on_valid_log(
        self,
        metrics: Dict[str, Any],
        metric_means: Dict[str, float],
        checkpoint_file_name: str,
        tasks_data: List[Any],
        scalar_name_to_total_experiences_key: Dict[str, str],
        step: int,
        **kwargs,
    ) -> None:
        """Log the validation metrics to wandb."""

        self._define_missing_metrics(
            metric_means=metric_means,
            scalar_name_to_total_experiences_key=scalar_name_to_total_experiences_key,
        )

        wandb.log(
            {
                **metric_means,
                "training_step": step,
            }
        )

    def _filter_tasks_for_video_logging(
        self,
        metrics: List[Dict[str, Any]],
        tasks_data: List[Any],
        apply_device_filter: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[Any]]:
        if tasks_data is None:
            return [], []

        filtered_metrics = []
        filtered_tasks = []

        for metric, task_data in zip(metrics, tasks_data):
            if not isinstance(task_data, dict):
                continue
            sensor_data = task_data.get("local_logging_callback_sensor")
            if sensor_data is None:
                continue
            if sensor_data.get("observations") is None:
                continue
            if (
                apply_device_filter
                and
                self.log_videos_from_device is not None
                and sensor_data.get("sampler_device") != self.log_videos_from_device
            ):
                continue
            filtered_metrics.append(metric)
            filtered_tasks.append(task_data)

            if len(filtered_metrics) >= self.max_videos_per_log:
                break

        return filtered_metrics, filtered_tasks

    def get_table_content(self, metrics, tasks_data, frames_with_logit_flag=False):
        sensor_data_list = [
            tasks_data[i]["local_logging_callback_sensor"]
            for i in range(len(tasks_data))
        ]

        observation_list = [
            sd["observations"] for sd in sensor_data_list
        ]

        list_of_video_frames = [self.combine_rgb_across_episode(obs) for obs in observation_list]

        path_list = [
            sd["path"] for sd in sensor_data_list
        ]

        if frames_with_logit_flag:
            frames_with_logits_list_numpy = [
                sd.get("frames_with_logits") for sd in sensor_data_list
            ]

        table_content = []
        frames_with_logits_list = []

        videos_without_logits_list = []

        for idx, (frames_without_logits, path, sensor_data) in enumerate(
            zip(list_of_video_frames, path_list, sensor_data_list)
        ):
            task_info = sensor_data.get("task_info", {})
            wandb_data = (
                wandb.Video(
                    np.moveaxis(np.array(frames_without_logits), [0, 3, 1, 2], [0, 1, 2, 3]),
                    fps=10,
                    format="mp4",
                ),
                wandb.Image(path[0]),
                sensor_data.get("ep_length", len(frames_without_logits)),
                sensor_data.get("success", False),
                sensor_data.get("dist_to_target", -1),
                task_info.get("task_type", ""),
                task_info.get("house_name", task_info.get("house_index", "")),
                task_info.get("target_object_type", ""),
                task_info.get("natural_language_spec", ""),
                task_info.get("id", f"unknown_{idx}"),
                idx,
            )

            videos_without_logits_list.append(
                wandb.Video(
                    np.moveaxis(np.array(frames_without_logits), [0, 3, 1, 2], [0, 1, 2, 3]),
                    fps=5,
                    format="mp4",
                ),
            )

            table_content.append(wandb_data)
            if frames_with_logit_flag:
                frames_with_logits_list.append(
                    wandb.Video(np.array(frames_with_logits_list_numpy[idx]), fps=10, format="mp4")
                )

        return table_content, frames_with_logits_list, videos_without_logits_list

    def on_test_log(
        self,
        checkpoint_file_name: str,
        metrics: Dict[str, Any],
        metric_means: Dict[str, float],
        tasks_data: List[Any],
        scalar_name_to_total_experiences_key: Dict[str, str],
        step: int,
        **kwargs,
    ) -> None:
        """Log the test metrics to wandb."""

        self._define_missing_metrics(
            metric_means=metric_means,
            scalar_name_to_total_experiences_key=scalar_name_to_total_experiences_key,
        )

        metrics_list, tasks_list = self._filter_tasks_for_video_logging(
            metrics=metrics.get("tasks", []),
            tasks_data=tasks_data,
            apply_device_filter=False,
        )

        if len(metrics_list) > 0:
            frames_with_logits_flag = False

            if "frames_with_logits" in tasks_list[0]["local_logging_callback_sensor"]:
                frames_with_logits_flag = True

            (
                table_content,
                frames_with_logits_list,
                videos_without_logit_list,
            ) = self.get_table_content(metrics_list, tasks_list, frames_with_logits_flag)

            video_dict = {"all_videos": {}}

            for vid, data in zip(frames_with_logits_list, table_content):
                idx = data[-1]
                video_dict["all_videos"][idx] = vid

            table = wandb.Table(
                columns=[
                    "Trajectory",
                    "Path",
                    "Episode Length",
                    "Success",
                    "Dist to target",
                    "Task Type",
                    "House Name",
                    "Target Object Type",
                    "Prompt",
                    "Task Id",
                    "Index",
                ]
            )

            for data in table_content:
                table.add_data(*data)

            wandb.log(
                {
                    **metric_means,
                    "training_step": step,
                    "Qualitative Examples": table,
                    "Videos": video_dict,
                }
            )
        else:
            wandb.log(
                {
                    **metric_means,
                    "training_step": step,
                    # "Qualitative Examples": table,
                    # "Videos": video_dict,
                }
            )

    def after_save_project_state(self, base_dir: str) -> None:
        pass

    def callback_sensors(self) -> Optional[Sequence[Sensor]]:
        return [
            WandbLoggingSensor(
                uuid="local_logging_callback_sensor", observation_space=gym.spaces.Discrete(1)
            ),
        ]
