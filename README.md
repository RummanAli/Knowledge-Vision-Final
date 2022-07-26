# KnowledgeVision
Fusion of data and knowledge into neural networks.

## Repo Info
- Knowledge Model: [Model](https://github.com/clcarwin/sphereface_pytorch/blob/master/model/sphere20a_20171020.7z)
- Resnet: [Model](https://github.com/clcarwin/sphereface_pytorch)
- Dermnet: [Model](https://github.com/clcarwin/sphereface_pytorch)

## Dataset
- G1020-data: [g1020-link]()
- Dermnet: [dermnet]()
- ISIC-2019: [isic-2019]()

## training
- python trainer.py --data_dir 'g1020_polygons' --dataset g1020 --model resnet --num_classes 2 --batch_size 16 --input_size 224
- python trainer.py --data_dir 'g1020_polygons' --dataset g1020 --model densenet --num_classes 2 --batch_size 16 --input_size 224
- python trainer.py --data_dir 'g1020_polygons' --dataset g1020 --model knowledge --num_classes 2 --batch_size 16 --input_size 224

## Dependencies
- Pytroch
- Pillow
- Sklearn
- OpenCV
- MTCNN
- ArgParse
