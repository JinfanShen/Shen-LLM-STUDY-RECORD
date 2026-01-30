# STUDY-RECORD INTRODUCTION
个人的学习大模型的学习笔记和资料保存。

学习内容：
## Attenion
--_MQA_GQA.ipynb:包含基本多头自注意力（MHA）、多查询注意力（MQA）和分组查询自注意力（GQA）。

--_sliding_Window.ipynb:包含滑窗注意力及和全局注意力一起的混合注意力机制。

参考资料：
1. https://blog.csdn.net/shizheng_Li/article/details/145809397  

## RoPE编码
--RoPE.ipynb:包含RoPE编码的实现。

--RoPE_Explanation.ipynb:解释了为什么实现RoPE编码时是按前后各一半来转换的，而不是论文里一对一对配对旋转来实现的。

参考资料：
1. https://yuanchaofa.com/post/hands-on-rope-position-embedding.
## Transformers
Transformers库的基本使用：

--model.ipynb:使用transformers库的model方法。

--pipeline.ipynb:使用transformers库的pipeline方法。

--tokenizer.ipynb:使用transformers库的tokenizer方法。

--classifier_demo.ipynb:使用transformers库实现基础的文本分类训练推理过程。

--Dataset.ipynb:使用Datasets库方法。

--load_cmrc2018.py：读取cmrc2018数据集的实现。Datasets库新版本不支持scripts,所以在该文件中尝试from_list导入。

--Evaluate.py:使用evaluate库的方式。

--Trainer.py: 使用transformers.Trainer训练之前的demo
参考资料或使用的数据集：
1. https://www.bilibili.com/video/BV18T411t7h6
2. https://github.com/SophonPlus/ChineseNlpCorpus
3. https://huggingface.co/docs/transformers/main/en/index
4. https://huggingface.co/datasets/madao33/new-title-chinese
5. cmrc2018数据集是从第一个B站UP主github下载 我也没找到原始地点
6. https://huggingface.co/docs/transformers/trainer

## Transformers_NLP
基于Transformers的NLP实现示例：

--Transformers显存优化.ipynb:优化显存的4种方式。

--命名实体识别任务.ipynb: 实现NER任务

--机器阅读理解.ipynb: 用截断处理实现阅读理解任务

--机器阅读理解的滑动窗口实现.ipynb: 用滑动窗口实现阅读理解任务

--多项选择任务.ipynb: 实现多项选择任务

-- 文本相似度.ipynb: 基于交互策略实现文本相似度任务

参考资料或使用的数据集：
1. https://www.bilibili.com/video/BV18T411t7h6
2. https://github.com/CLUEbenchmark/SimCLUE/tree/main?tab=readme-ov-file
   


## MOE
MOE -> Sparse MOE -> ShareSparse MOE原理和实现; 已经负载均衡损失函数的原理和实现    
--MOE_model.ipynb: MOE原理和实现  
--MOE学习笔记.md ：AI生成  
--auxiliary_loss.py : 负载均衡损失函数的实现和逐步参数调试  
--MOE负载均衡损失学习笔记.md ：AI生成  
 
参考资料：
1. https://wnma3mz.github.io/hexo_blog/2024/06/15/MoE%E4%B8%AD%E8%B4%9F%E8%BD%BD%E5%9D%87%E8%A1%A1Loss%E5%AE%9E%E7%8E%B0%E5%AF%B9%E6%AF%94/
2. https://www.bilibili.com/video/BV1ZbFpeHEYr/?spm_id_from=333.1391.0.0&vd_source=836e607a53e03229d373ac56cd1b7a88
3. [Switch Transformer](https://arxiv.org/abs/2101.03961)