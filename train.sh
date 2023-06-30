CUDA_VISIBLE_DEVICES=0 python3 train_cnn.py --training_title miccai23_train_resnet18_isic --epochs 200 --lr 0.01 --batch_size 128 --dataset isic2019 --model resnet18 --class_num 8 
CUDA_VISIBLE_DEVICES=0 python3 train_cnn_me.py --training_title miccai23_train_me_resnet18_isic --epochs 200 --lr 0.01 --batch_size 128 --dataset isic2019 --model me_resnet18 --class_num 8 
CUDA_VISIBLE_DEVICES=0 python3 train_cnn.py --training_title miccai23_train_vgg11_fitzpatrick17k --epochs 200 --lr 0.01 --batch_size 128 --dataset fitzpatrick17k --model vgg11 --class_num 114 
CUDA_VISIBLE_DEVICES=0 python3 train_cnn_me.py --training_title miccai23_train_me_vgg11_fitzpatrick17k --epochs 200 --lr 0.01 --batch_size 128 --dataset fitzpatrick17k --model me_vgg11 --class_num 114 

CUDA_VISIBLE_DEVICES=0 python3 train_hsic.py --training_title miccai23_train_hsic_isic --epochs 200 --lr 0.01 --batch_size 128 --dataset isic2019 --model hsic_resnet18 --class_num 8 
CUDA_VISIBLE_DEVICES=0 python3 train_hsic_me.py --training_title miccai23_train_me_hsic_isic --epochs 200 --lr 0.01 --batch_size 128 --dataset isic2019 --model me_hsic_resnet18 --class_num 8 
CUDA_VISIBLE_DEVICES=0 python3 train_hsic.py --training_title miccai23_train_hsic_fitz --epochs 200 --lr 0.01 --batch_size 128 --dataset fitzpatrick17k --model hsic_vgg11 --class_num 114
CUDA_VISIBLE_DEVICES=0 python3 train_hsic_me.py --training_title miccai23_train_me_hsic_fitz --epochs 200 --lr 0.01 --batch_size 128 --dataset fitzpatrick17k --model me_hsic_vgg11 --class_num 114

CUDA_VISIBLE_DEVICES=0 python3 train_mfd.py --training_title miccai23_train_mfd_isic_train_teacher --epochs 200 --lr 0.01 --batch_size 128 --dataset isic2019 --model mfd_resnet18 --class_num 8 --train_teacher True
CUDA_VISIBLE_DEVICES=0 python3 train_mfd.py --training_title miccai23_train_mfd_isic --epochs 200 --lr 0.01 --batch_size 128 --dataset isic2019 --model mfd_resnet18 --class_num 8 --teacher_path ./networks/121/miccai23_train_mfd_isic_train_teacher/5.pth
CUDA_VISIBLE_DEVICES=0 python3 train_mfd_me.py --training_title miccai23_train_me_mfd_isic_train_teacher --epochs 200 --lr 0.01 --batch_size 128 --dataset isic2019 --model me_mfd_resnet18 --class_num 8 --train_teacher True 
CUDA_VISIBLE_DEVICES=0 python3 train_mfd_me.py --training_title miccai23_train_me_mfd_isic --epochs 200 --lr 0.01 --batch_size 128 --dataset isic2019 --model me_mfd_resnet18 --class_num 8 --teacher_path ./networks/121/miccai23_train_me_mfd_isic_train_teacher/5.pth --train_teacher False 
CUDA_VISIBLE_DEVICES=0 python3 train_mfd.py --training_title miccai23_train_mfd_fitz_train_teacher --epochs 200 --lr 0.01 --batch_size 128 --dataset fitzpatrick17k --model mfd_vgg11 --class_num 114 --train_teacher True
CUDA_VISIBLE_DEVICES=0 python3 train_mfd.py --training_title miccai23_train_mfd_fitz --epochs 200 --lr 0.01 --batch_size 128 --dataset fitzpatrick17k --model mfd_vgg11 --class_num 114 --teacher_path ./networks/121/miccai23_train_mfd_fitz_train_teacher/5.pth
CUDA_VISIBLE_DEVICES=0 python3 train_mfd_me.py --training_title miccai23_train_me_mfd_fitz_train_teacher --epochs 200 --lr 0.01 --batch_size 128 --dataset fitzpatrick17k --model me_mfd_vgg11 --class_num 114 --train_teacher True 
CUDA_VISIBLE_DEVICES=0 python3 train_mfd_me.py --training_title miccai23_train_me_mfd_fitz --epochs 200 --lr 0.01 --batch_size 128 --dataset fitzpatrick17k --model me_mfd_vgg11 --class_num 114 --teacher_path ./networks/121/miccai23_train_me_mfd_fitz_train_teacher/5.pth --train_teacher False
