
from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['settings.toml', '.secrets.toml'],
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.

# Whether to generate the output video with the detection results
generate_detection_output_video = settings.get('generate_detection_output_video', False)