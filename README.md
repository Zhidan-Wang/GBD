# Toward Bias-Agnostic Recommender Systems: A Universal Generative Framework(GBD)
## 1. Abstract
User behavior data, such as ratings and clicks, has been widely used to build personalizing models for recommender systems. However, many unflattering factors (e.g., popularity, ranking position, users' selection) significantly affect the performance of the learned recommendation model. Most existing work on unbiased recommendation addressed these biases from sample granularity (e.g., sample reweighting, data augmentation) or from the perspective of representation learning (e.g., bias-modeling).
However, these methods are usually designed for a specific bias, lacking the universal capability to handle complex situations where multiple biases co-exist. Besides, rare work frees itself from laborious and sophisticated debiasing configurations (e.g., propensity scores, imputed values, or user behavior-generating process).

Towards this research gap, in this paper, we propose a universal **G**enerative framework for **B**ias **D**isentanglement termed as **GDB**, constantly generating calibration perturbations for the intermediate representations during training to keep them from being affected by the bias. Specifically, a bias-identifier that tries to retrieve the bias-related information from the representations is first introduced. Then the calibration perturbations are generated to significantly deteriorate the bias-identifier's performance, making the bias gradually disentangled from the calibrated representations. 
Therefore, without relying on notorious debiasing configurations, a bias-agnostic model is obtained under the guidance of the bias identifier.
We further present its universality by subsuming the representative biases and their mixture under the proposed framework. Finally, extensive experiments on the real-world, synthetic, and semi-synthetic datasets have demonstrated the superiority of the proposed approach against a wide range of recommendation debiasing methods.

## 2. Overall framework
<img src='https://user-images.githubusercontent.com/31196524/151699502-ac6b2484-274e-4074-8ee9-9bbe43cf69af.png' width="80%">

## 3. Runtime Environment

* System:Linux dell-PowerEdge-R730

* CPU: Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz

* CPU-Memory:25G

* GPU:NVIDIA Corporation GV100 [TITAN V] (rev a1)

* GPU-Memory:12G

* Pytorch: 1.7.0

* CUDA:10.1

## 4. Usage
### 4.1 Run ADS_S, ADS_Pos (selection bias/ position bias)
Go the ads/ directory and run the following command:

- For dataset Yahoo!R3:

```shell
python ADS_S.py --dataset yahooR3
```

- For dataset Coat:

```shell
python ADS_S.py --dataset coat
```

- For dataset Simulation:

```shell
python ADS_Pos.py --dataset simulation
```
### 4.2 Run ADS_P (popularity)
Go the ads_p/ directory 

(1) Configure some training parameters through the **test.yaml**.

```shell
Set **mode = 1**, **norm = 1e-4**.
```

(2) Run the following command:

```shell
python run_recbole.py --dataset ml-1m --model BPR --config_files test.yaml
```

