import os
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from Model_Class import Classifier_AMS, Classifier_AMS_SAP_MultiHead

run_in_background = False # make tqdm to be silent mode
show_model_summary = False # Whether to print the model summary. You may set it to False if you don't want to see the model summary.
allow_device = False  # Set to False if you want to force using CPU.
use_pin_memory = False  # Set to True if you use GPU. False for CPU and MPS.
n_save_step = 10000 # The number of steps for saving the model. You may adjust it based on your needs.
ratio_train = 0.9 # 90% of the dataset are allocated into training set. 10% of them go to validation set. 
ams_m = 0.25 # Margin for AM-Softmax. You may adjust it based on your needs.
ams_s = 30.0 # Scale for AM-Softmax. You may adjust it based on your needs.
n_transformer_layer = 1 # The number of transformer layers in the classifier. 
n_head_SAP = 5 # The number of heads in self-attentive pooling. 
# Batch size for training, validation, and testing.
# A greater batch size usually gives a more stable gradient.
batch_size = 1
# 0 means only the main process will load data. Greater than 0 means number of subprocesses to use for data loading.
# If you use cuda, you may set it to a greater value like 4 or 8 to accelerate data loading.
num_workers = 0  # You may change this value based on your system configuration.
file_path = "./project_data_voice" # the location of the dataset
model_path = "./specker_classifier_v3_1_layer_total_step_140000_step_140000_acc_0.91435_mulihead_sap.pth"

class InferenceDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        # Load the mapping from speaker neme to their corresponding id.
        mapping_path = Path(data_dir) / "mapping.json"
        mapping = json.load(mapping_path.open())
        self.speaker2id = mapping["speaker2id"]

        # Load metadata of training data.
        metadata_path = Path(data_dir) / "metadata.json"
        metadata = json.load(open(metadata_path))["speakers"]

        # Get the total number of speaker.
        self.speaker_num = len(metadata.keys())
        self.data = []
        for speaker in metadata.keys():
            for utterances in metadata[speaker]:
                self.data.append([utterances["feature_path"], self.speaker2id[speaker]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feat_path, speaker = self.data[index]
        # Load preprocessed mel-spectrogram.
        mel = torch.load(os.path.join(self.data_dir, feat_path))
        mel = torch.FloatTensor(mel)
        speaker = torch.FloatTensor([speaker]).long()
        return mel, speaker

    def get_speaker_number(self):
        return self.speaker_num


def inference_collate_batch(batch):
    """Collate a batch of data."""
    mel, speaker = zip(*batch)     # each is tuple length B # we assurm B is 1.
    mel = mel[0].float()           # (L, 40) if batch_size=1
    speaker = speaker[0].view(-1).long()  # (1,)
    return mel.unsqueeze(0), speaker       # mel -> (1, L, 40), speaker -> (1,)


# Automatically choose the device to use.
if allow_device:
# Move the network to the appropriate device
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
        print("Using GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
        num_workers = 0 # MPS does not support multi-process data loading, so we set num_workers to 0.
        use_pin_memory = False # MPS does not support pin_memory, so we set it to False.
    else:
        device = torch.device("cpu")
        print("Using CPU")
        use_pin_memory = False
        num_workers = 0 # You do not multiple process to move data when using CPU, so we set num_workers to 0.
else:
    device = torch.device("cpu")
    print("Using CPU")
    use_pin_memory = False
    num_workers = 0 # You do not multiple process to move data when using CPU, so we set num_workers to 0.

dataset = InferenceDataset(file_path)
speaker_num = dataset.get_speaker_number()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=inference_collate_batch)
print(f"[Info]: Finish loading data!")
model = Classifier_AMS_SAP_MultiHead(n_spks=speaker_num, am_m=ams_m, am_s=ams_s, n_transformer_layer=n_transformer_layer, n_head=n_head_SAP).to(device)
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
model.eval()
batch_accs = []
with torch.no_grad():
    for batch in tqdm(dataloader, ncols=0, desc="Inference", unit=" step", disable=run_in_background):
        mels, labels = batch
        mels = mels.to(device)
        labels = labels.to(device)
        logits = model(mels)
        pred = torch.argmax(logits, dim=1)
        acc = (pred == labels.view(-1)).float().mean().item()
        batch_accs.append(acc)
valid_acc = sum(batch_accs) / len(batch_accs)
print(f"[Info]: Finish inference! Validation accuracy: {valid_acc:.5f}")



