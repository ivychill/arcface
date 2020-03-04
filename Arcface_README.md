# 快速开始：
    登录80服务器： 
    用户名：zb-test 
    密码：123456

## 键入以下环境变量的命令（会在每一次重启时失效）：
    export PATH=$PATH:/usr/local/cuda-10.0/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
    export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-10.0/lib64
    source /etc/profile

### 数据集的准备：（选择快速开始请略过）
    下载faces_emore.zip
    解压文件到InsightFace_pytorch_one(或者_two)的data文件下（里面已经有了）
    one和two的项目的用处在于one用于单GPU的训练与单GPU的测试，two目前仅仅用于多GPU的训练

    运行以下命令可以将压缩文件转为可以训练与测试的文件：
        cd InsightFace_one
        python3 prepare.py

### 训练：
    注意：部分代码使用了绝对路径，在报错时可自行修改
    已经踩过很多坑，坑在80服务器上的repo中基本填完。部分填过的坑会在这份说明最后列出。

    如果你想要的进行单GPU训练：
        cd InsightFace_pytorch_one
        python3 train.py -net mobilefacenet

    或者你想要单机多卡训练
        cd InsightFace_pytorch_two
        CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py -net mobilefacenet
        解释：  CUDA_VISIBLE_DEVICES= 指定可见的GPU，用来训练的
                nproc_per_node= 值设定为想使用的GPU的个数

### 测试：（目前只支持单GPU测试单GPU训练的模型与多GPU训练的模型，测试时存在区别）
    如果你想测试work_space/models文件夹下保存的文件，这时只需要将model*.pth文件 cp 到save文件夹下，你可以多拷贝几个你认为有竞争力acc的模型。
    （注意，部分模型可能在训练的时候保存的不完整可能会出现无法加载的情况，建议多用几个模型测试）
    运行：
        python3 eval.py 
    默认测试2个模型是用多GPU训练得到的，模型的名称默认是 model_ir_se50.pth 与 model_mobilefacenet.pth
    如果你想测试单GPU训练的模型，请在"/home/test/InsightFace_Pytorch_one/Learner.py"中注释掉99行，并反注释100行
    原因在于多卡保存的模型中变量的名称会比单卡训练的多出七个字符 "module."，而99行中使用的load_network函数就是用来删除这些字符的，
    如果测试的是单卡训练保存的模型则不需要删除。

### Q&A:
    问题1：数据集，训练与测试的相关文件与配置文件。
        
    答：
        与数据集关系密切的文件有prepare.py(用于训练集与测试集的生成)，data/data_pipe.py（数据集的载入）
        训练 train.py 用于加载配置与解析args，真的train函数在Learn.py中
        验证，训练设置每1/10个epoch会验证一次，evaluate函数在Learn.py中，可以自行修改验证频率
        测试 eval.py

    问题2：训练过程中模型为什么会在work_space/models文件夹下保存多次 三个.pth文件？
        例如这样：    
            head*.pth (保存的是损失函数部分的参数)
            model*.pth （保存的是模型的参数***比较重要***）
            optimizer*.pth （保存的是优化器部分的参数）

    答：
        因为训练过程中（见Learn.py中的train与evaluate）设定每1/10个epoch进行一次验证，符合条件会则
        会被保存，保存的参数的意义参考上述。
    
    问题3：出现了File "/home/test/.local/lib/python3.5/site-packages/torch/tensor.py", line 451, in __array__
                return self.numpy().astype(dtype, copy=False)
                TypeError: can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
    
    答：
        pytorch或者python版本错误，建议在出错位置  l2_norm(emb_batch) 后面加上.cpu()

    问题4：训练的模型无法load

    答：
        （1）检查训练的模型是单GPU训练的模型还是多GPU训练的模型，如果是单GPU训练的模型，在"/home/test/InsightFace_Pytorch_one/Learner.py"中注释掉99行，并反注释100行
        反之，保留99行，注释100行
        （2）此模型保存时候出错，多拷贝几个保存的模型试试
    
    问题5：我从InsightFace_pytorch的Github中下载的repo，在准备数据，训练，测试时出现了问题怎么办？

    答：
        数据集准备与载入：
            单GPU数据集准备与载入建议参考 "/home/test/InsightFace_Pytorch_one"
            单机多卡数据集准备与载入建议参考 "/home/test/InsightFace_Pytorch_two"
        训练：
            单GPU训练建议参考 "/home/test/InsightFace_Pytorch_one"
            单机多卡训练建议参考 "/home/test/InsightFace_Pytorch_two"
        测试：
            目前仅参考 单机多卡训练建议参考 "/home/test/InsightFace_Pytorch_one"

 ### 部分可能有用的BUG日志
    BUG1:
        File "/home/test/.local/lib/python3.5/site-packages/torch/tensor.py", line 451, in __array__
        return self.numpy().astype(dtype, copy=False)
        TypeError: can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
    SOLUTION:
        见问题三，如果搞不定，终极建议
        将tensor.py的451行改为：
	    return self.cpu().numpy().....
    

 #########################

    BUG2:

    File "eval.py", line 12, in <module>
    vgg2_fp, vgg2_fp_issame = get_val_pair(conf.emore_folder, 'vgg2_fp')
    File "/home/test/InsightFace_Pytorch_one/data/data_pipe.py", line 70, in get_val_pair
    carray = bcolz.carray(rootdir = path/name, mode='r')
    File "bcolz/carray_ext.pyx", line 1067, in bcolz.carray_ext.carray.__cinit__
    File "bcolz/carray_ext.pyx", line 1367, in bcolz.carray_ext.carray._read_meta
    File "/usr/lib/python3.5/posixpath.py", line 89, in join
    genericpath._check_arg_types('join', a, *p)
    File "/usr/lib/python3.5/genericpath.py", line 143, in _check_arg_types
    (funcname, s.__class__.__name__)) from None
    TypeError: join() argument must be str or bytes, not 'PosixPath'

    ###---SOLUTION---###

    str(rootdir) --->OK eval.py
#########################

    BUG3:
    
    Traceback (most recent call last):
    File "train.py", line 30, in <module>
    learner = face_learner(conf)
    File "/home/test/InsightFace_Pytorch_one/Learner.py", line 29, in __init__
    self.loader, self.class_num = get_train_loader(conf)
    File "/home/test/InsightFace_Pytorch_one/data/data_pipe.py", line 47, in get_train_loader
    ds, class_num = get_train_dataset(conf.emore_folder/'imgs')
    File "/home/test/InsightFace_Pytorch_one/data/data_pipe.py", line 24, in get_train_dataset
    ds = ImageFolder(imgs_folder, train_transform)
    File "/home/test/.local/lib/python3.5/site-packages/torchvision/datasets/folder.py", line 209, in __init__
    is_valid_file=is_valid_file)
    File "/home/test/.local/lib/python3.5/site-packages/torchvision/datasets/folder.py", line 93, in __init__
    classes, class_to_idx = self._find_classes(self.root)
    File "/home/test/.local/lib/python3.5/site-packages/torchvision/datasets/folder.py", line 122, in _find_classes
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    TypeError: scandir: illegal type for path parameter

    line 24 改为str
#########################

    BUG4:

    Traceback (most recent call last):
    File "train.py", line 30, in <module>
    learner = face_learner(conf)
    File "/home/test/InsightFace_Pytorch_one/Learner.py", line 57, in __init__
    self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame,     self.cfp_fp_issame, self.lfw_issame = get_val_data  (self.loader.dataset.root.parent)
    AttributeError: 'str' object has no attribute 'parent'

    solution：
    将self.loader.dataset.root改为Path类型，需要import pathlib     
 ########################
    
    修改日志1：
        data_pipe.py
        71 行增加了rootdir = path/name
        72 行的rootdir转为了str类型

    修改日志2：
        将self.loader.dataset.root改为Path类型，需要import pathlib  