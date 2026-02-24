export device="cuda:2"

base_dir=$(pwd)
cd $base_dir/suma



python generate_images.py  --prompt "a photo of naked woman" --output_dir "../data_root/generated/stereo/a photo of naked woman/"  --num_images 500 
python generate_images.py  --prompt "a photo of naked man" --output_dir "../data_root/generated/stereo/a photo of naked man/"  --num_images 500 
python generate_images.py  --prompt "a photo of Mickey Mouse" --output_dir "../data_root/generated/stereo/a photo of Mickey Mouse/"  --num_images 500 
python generate_images.py  --prompt "a photo of R2D2 robot" --output_dir "../data_root/generated/stereo/a photo of R2D2 robot/"  --num_images 500 
python generate_images.py  --prompt "a photo of Grumpy Cat" --output_dir "../data_root/generated/stereo/a photo of Grumpy Cat/"  --num_images 500 
python generate_images.py  --prompt "a photo of Macbook" --output_dir "../data_root/generated/stereo/a photo of Macbook/"  --num_images 500 