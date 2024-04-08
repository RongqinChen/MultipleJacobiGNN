# for dataset in low high rejection band comb
# do
#     for model in jacobi legendre
#     do
#         python ImgFilter.py --optruns 100 --$model --dataset $dataset --path results/ --name $model+$dataset | tee results/$model+$dataset.out
#     done
# done

model=legendre

dataset1=low
dataset2=high
dataset3=rejection
dataset4=band
dataset5=comb


(CUDA_VISIBLE_DEVICES=0 python ImgFilter.py --optruns 100 --$model --dataset $dataset1 --path results/ --name $model+$dataset1 | tee results/$model+$dataset1.out)
# wait
(CUDA_VISIBLE_DEVICES=0 python ImgFilter.py --optruns 100 --$model --dataset $dataset2 --path results/ --name $model+$dataset2 | tee results/$model+$dataset2.out)
# wait
(CUDA_VISIBLE_DEVICES=0 python ImgFilter.py --optruns 100 --$model --dataset $dataset3 --path results/ --name $model+$dataset3 | tee results/$model+$dataset3.out)
# wait
(CUDA_VISIBLE_DEVICES=0 python ImgFilter.py --optruns 100 --$model --dataset $dataset4 --path results/ --name $model+$dataset4 | tee results/$model+$dataset4.out)
# wait
(CUDA_VISIBLE_DEVICES=0 python ImgFilter.py --optruns 100 --$model --dataset $dataset5 --path results/ --name $model+$dataset5 | tee results/$model+$dataset5.out)
# wait
