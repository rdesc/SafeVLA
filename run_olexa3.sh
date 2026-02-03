module load apptainer/1.4.5

sed -e 's|/usr/lib64/libGLX_nvidia.so.0|/.singularity.d/libs/libEGL_nvidia.so.0|g' \
    /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json > \
    ${SLURM_TMPDIR}/nvidia_icd.x86_64.json
sed -e 's|libGLX_nvidia.so.0|/.singularity.d/libs/libEGL_nvidia.so.0|g' \
    /usr/share/vulkan/implicit_layer.d/nvidia_layers.json > \
    ${SLURM_TMPDIR}/nvidia_layers.json

APP_BIND_FLAGS=(
    -B${SLURM_TMPDIR}:/run/user:rw
    -B${SLURM_TMPDIR}/nvidia_icd.x86_64.json:/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json:ro
    -B${SLURM_TMPDIR}/nvidia_layers.json:/usr/share/vulkan/implicit_layer.d/nvidia_layers.json:ro
)
DRV_VER="$RSNT_CUDA_DRIVER_VERSION"
for libs in /usr/lib64/libnvidia-allocator.so.1               \
            /usr/lib64/libnvidia-api.so.1                     \
            /usr/lib64/libnvidia-eglcore.so.${DRV_VER}        \
            /usr/lib64/libnvidia-egl-gbm.so.1                 \
            /usr/lib64/libnvidia-egl-xcb.so.1                 \
            /usr/lib64/libnvidia-egl-xlib.so.1                \
            /usr/lib64/libnvidia-glcore.so.${DRV_VER}         \
            /usr/lib64/libnvidia-glsi.so.${DRV_VER}           \
            /usr/lib64/libnvidia-glvkspirv.so.${DRV_VER}      \
            /usr/lib64/libnvidia-gpucomp.so.${DRV_VER}        \
            /usr/lib64/libnvidia-gtk3.so.${DRV_VER}           \
            /usr/lib64/libnvidia-present.so.${DRV_VER}        \
            /usr/lib64/libnvidia-rtcore.so.${DRV_VER}         \
            /usr/lib64/libnvidia-sandboxutils.so.1            \
            /usr/lib64/libnvidia-tls.so.${DRV_VER}            \
            /usr/lib64/libnvidia-vksc-core.so.1               \
            /usr/lib64/libnvidia-wayland-client.so.${DRV_VER} \
            ; do
    APP_BIND_FLAGS+=("-B${libs}:/.singularity.d/libs/$(basename "${libs}"):ro")
done

apptainer exec --nv -e ${APP_BIND_FLAGS[@]} --env XDG_RUNTIME_DIR=/run/user safevla.sif \
  python3 -c "from ai2thor.platform import CloudRendering; from ai2thor.controller import Controller; print('imported pkgs, launching controller....'); controller = Controller(platform=CloudRendering); print('launched controller'); controller.step(dict(action='Initialize', gridSize=0.25)); print('stepped with controller')"

