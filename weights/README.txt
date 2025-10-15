These are the current weights for trained YOLO bee detection models.

The current best practice is to use '2000-lvl-model.pt' - this gives good results.

The larger model trained on 4000 images ('4000-lvl-model.pt') has seen a broader range of data, and so will drop detections less frequently. 
However, this model does introduce a lot more false positives, which will cause our tracking logic to default to missed detection cases more often -
which overall will have a poorer tracking accuracy.

The other files in this folder are older models, and I don't really recommend you use them.