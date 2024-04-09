# for dataset in low high rejection band comb
# do
#     for model in jacobi legendre
#     do
#         python ImgMultipleFilter.py --optruns 100 --$model --dataset $dataset --path results/ --name m$model+$dataset | tee results/m$model+$dataset.out
#     done
# done

model=jacobi

dataset1=low
dataset2=high
dataset3=rejection
dataset4=band
dataset5=comb


(CUDA_VISIBLE_DEVICES=0 python ImgMultipleFilter.py --optruns 100 --$model --dataset $dataset1 --path results/ --name m$model+$dataset1 | tee results/m$model+$dataset1.out)
# wait
(CUDA_VISIBLE_DEVICES=0 python ImgMultipleFilter.py --optruns 100 --$model --dataset $dataset2 --path results/ --name m$model+$dataset2 | tee results/m$model+$dataset2.out)
# wait
(CUDA_VISIBLE_DEVICES=0 python ImgMultipleFilter.py --optruns 100 --$model --dataset $dataset3 --path results/ --name m$model+$dataset3 | tee results/m$model+$dataset3.out)
# wait
(CUDA_VISIBLE_DEVICES=0 python ImgMultipleFilter.py --optruns 100 --$model --dataset $dataset4 --path results/ --name m$model+$dataset4 | tee results/m$model+$dataset4.out)
# wait
(CUDA_VISIBLE_DEVICES=0 python ImgMultipleFilter.py --optruns 100 --$model --dataset $dataset5 --path results/ --name m$model+$dataset5 | tee results/m$model+$dataset5.out)
# wait
