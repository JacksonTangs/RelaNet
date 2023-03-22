# RelaNet

Most format inference methods need sequence alignment to deal with variable-length fields. However, it alignment quality doesn’t perform well. In this study, we address the problem of how to mitigate the impact from variable-length fields in inferring protocol format specifications. We propose a novel relational reasoning-based approach RelaNet to learn the **context relations** between keywords. This type of relation can be used to diminish the impacts of variable-length fields, which would normally strongly impair the quality of protocol format inference.

## Requirements

- Python 3.x
- PyTorch 1.x
- NumPy
- SciPy
- Joblib
- Pickle

## Installation

To install RelaNet, clone the repository and install the required dependencies using pip:

```bash
bashCopy codegit clone https://github.com/JacksonTangs/RelaNet.git
cd relanet
pip install -r requirements.txt
```



## Instruction & Structure

The whole project includes three phases, **Coarse structure generation**, **Relation learning**, and **Fine structure generation**.

RelaNet includes two datasets for training and evaluation:

- `RelaNet-DATA`: A folder containing `Dataset-I` and `Dataset-II`, each of them contain 5 protocol datasets (DNS, Modbus, DNP3, NTP, DHCP).

### Coarse structure generation

This phase includes the following components:

- `big_dic_make.py`: For creating a large dictionary of *n*-grams, which records all *n*-grams frequency .
- `dict-make.py`: Selecting pre-configured rank of *n*-grams for generating smaller dictionary (HF-Dictionary).
- `question-making.py`: Generating the question for each payloads.



To generate the coarse structure, run the Python scripts in the following order:

1. Making the large dictionary for each protocol, run 

   ```bash
   python big_dic_make.py --protocol_type <ProtocolName>
   ```

   

2. Selecting the section of the extensive dictionary that falls below the designated rank threshold of <RankThreshold>.

   ```bash
   python dict-make.py --protocol_type <ProtocolName> --dict <RankThreshold>
   ```

   

3. Run `question-making.py` to generate questions from the dataset for each payloads, <FuzzRange> and <RankThreshold> are preconfigured.

   ```bash
   python question-making.py --protocol_type <ProtocolName> --fuzz_range <FuzzRange> --dictionay_choose <RankThreshold>
   ```

   Up to here, the coarse structure is generated for each payloads.



### Relation learning

This phase includes the following components:

- `model/model_test.py`: Restore the main network of the relation learning phase used in training or testing phase.
- `main.py`: The main Python script that contains the code for training the neural network.
- `main_test.py`: The main Python script that contains the code for testing the neural network model.



To learn the relations in the coarse structure, run the Python scripts in the following order:

1. Training the model for corresponding <ProtocolName> in the setting of <FuzzRange> and <RankThreshold>, run 

   ```bash
   python main.py --dictionary_choose <RankThreshold> --fuzz_range <FuzzRange> --protocol_type <ProtocolName>
   ```

   

2. Extracting the relation model for each payloads in testing dataset under the question answer threshold <QuestionThreshold> ($p_{T}$ in article).

   ```bash
   python main_test.py --dic_rank <RankThreshold> --ans_prob_threshold <QuestionThreshold> --fuzz_range <FuzzRange> --protocol_type <ProtocolName>
   ```

   

### Fine structure generation

This phase includes the following components:

- `*_parse.py`: The files records the true structure of each protocol.
- `formats_inference.py`: Inference the fine structure of the payload.
- `inference_cluster.py`: Clustering the relation result produced in the previous phase.



For the purpose of generating the fine structure, run the Python scripts in the following order:

1. Clustering the relation result produced in the relation learning phase. <ProtocolName> in the setting of <FuzzRange> and <RankThreshold>, run 

   ```bash
   python inference_cluster.py --fuzz <FuzzRange> --dict <RankThreshold> --threshold <QuestionThreshold> --protocol <ProtocolName>
   ```

   

2. Extracting the structure of the first <MetricThreshold> bytes of the payload under predefined configuration. 

   ```bash
   python formats_inference.py --fuzz <FuzzRange> --dict <RankThreshold> --threshold <QuestionThreshold> --protocol <ProtocolName> --metric_threshold <MetricThreshold>
   ```

   

The result of fine structure matching true structure will be saved in a csv file in `final-result` folder.

## License

RelaNet is released under the [MIT License](https://opensource.org/licenses/MIT).