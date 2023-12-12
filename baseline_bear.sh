python launch.py --config configs/ours.yaml --train --gpu 0 data.dataroot="data/bear" data.camera_layout="front" data.camera_distance=1 data.eval_interpolation=[1,10,50] system.prompt_processor.prompt="Turn it into a panda" system.start_editing_step=1112 system.loss.lambda_sds=10.0 system.loss.lambda_zero123=10.0
python launch.py --config configs/ours.yaml --train --gpu 0 data.dataroot="data/bear" data.camera_layout="front" data.camera_distance=1 data.eval_interpolation=[1,10,50] system.prompt_processor.prompt="Turn it into a panda" system.start_editing_step=1112 system.loss.lambda_sds=0.0 system.loss.lambda_zero123=10.0
python launch.py --config configs/ours.yaml --train --gpu 0 data.dataroot="data/bear" data.camera_layout="front" data.camera_distance=1 data.eval_interpolation=[1,10,50] system.prompt_processor.prompt="Turn it into a panda" system.start_editing_step=1112 system.loss.lambda_sds=10.0 system.loss.lambda_zero123=0
python launch.py --config configs/instructnerf2nerf_right.yaml --train --gpu 0 data.dataroot="data/bear" data.camera_layout="front" data.camera_distance=1 data.eval_interpolation=[1,10,50] system.prompt_processor.prompt="Turn it into a panda" system.start_editing_step=612 system.loss.lambda_sds=0.0 system.loss.lambda_zero123=0
python launch.py --config configs/ours.yaml --train --gpu 0 data.dataroot="data/bear" data.camera_layout="front" data.camera_distance=1 data.eval_interpolation=[1,10,50] system.prompt_processor.prompt="Turn it into a panda" system.start_editing_step=1121 system.loss.lambda_sds=10.0 system.loss.lambda_zero123=10.0
python launch.py --config configs/ours.yaml --train --gpu 0 data.dataroot="data/bear" data.camera_layout="front" data.camera_distance=1 data.eval_interpolation=[1,10,50] system.prompt_processor.prompt="Turn it into a panda" system.start_editing_step=1121 system.loss.lambda_sds=0.0 system.loss.lambda_zero123=10.0
python launch.py --config configs/ours.yaml --train --gpu 0 data.dataroot="data/bear" data.camera_layout="front" data.camera_distance=1 data.eval_interpolation=[1,10,50] system.prompt_processor.prompt="Give it a hat" system.start_editing_step=1112 system.loss.lambda_sds=10.0 system.loss.lambda_zero123=10.0
python launch.py --config configs/ours.yaml --train --gpu 0 data.dataroot="data/bear" data.camera_layout="front" data.camera_distance=1 data.eval_interpolation=[1,10,50] system.prompt_processor.prompt="Give it a hat" system.start_editing_step=1112 system.loss.lambda_sds=0.0 system.loss.lambda_zero123=10.0
python launch.py --config configs/ours.yaml --train --gpu 0 data.dataroot="data/bear" data.camera_layout="front" data.camera_distance=1 data.eval_interpolation=[1,10,50] system.prompt_processor.prompt="Give it a hat" system.start_editing_step=1112 system.loss.lambda_sds=10.0 system.loss.lambda_zero123=0
python launch.py --config configs/instructnerf2nerf_right.yaml --train --gpu 0 data.dataroot="data/bear" data.camera_layout="front" data.camera_distance=1 data.eval_interpolation=[1,10,50] system.prompt_processor.prompt="Give it a hat" system.start_editing_step=612 system.loss.lambda_sds=0.0 system.loss.lambda_zero123=0
python launch.py --config configs/ours.yaml --train --gpu 0 data.dataroot="data/bear" data.camera_layout="front" data.camera_distance=1 data.eval_interpolation=[1,10,50] system.prompt_processor.prompt="Give it a hat" system.start_editing_step=1121 system.loss.lambda_sds=10.0 system.loss.lambda_zero123=10.0
python launch.py --config configs/ours.yaml --train --gpu 0 data.dataroot="data/bear" data.camera_layout="front" data.camera_distance=1 data.eval_interpolation=[1,10,50] system.prompt_processor.prompt="Give it a hat" system.start_editing_step=1121 system.loss.lambda_sds=0.0 system.loss.lambda_zero123=10.0