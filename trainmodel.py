import torch
import torch.profiler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from ModelAi.projectFeature import PJFTrainModel, PJF
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def misc_rate(step, model_size, factor, warmup_steps):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    step = 1 if step == 0 else step

    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
    )


class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_A = torch.load(self.data["A"][idx], map_location="cpu")
        data_B = torch.load(self.data["B"][idx], map_location="cpu")
        return data_A, data_B


if __name__ == "__main__":
    dataset = CustomDataset("Data\\trainProjection.csv")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=16)

    PJF_model = PJF()
    PJF_model.to("cuda")
    model = PJFTrainModel(model=PJF_model)
    model.to("cuda")

    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=1)
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: misc_rate(step, 3, factor=10, warmup_steps=3000 * 16),
    )

    num_epochs = 10

    for epoch in tqdm(range(num_epochs)):
        sum_loss = 0
        loss = 0
        step = 0
        for batch_idx, (data_a, data_b) in tqdm(enumerate(dataloader)):
            data_a = data_a.to("cuda").squeeze(0)
            data_b = data_b.to("cuda").squeeze(0)

            output = model(data_a, data_b)
            target = torch.zeros_like(output).to("cuda")
            loss += criterion(output, target)
            sum_loss += loss.item()
            step += 1
            if step == 256:
                loss /= step
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                step = 0
                loss = 0
        if step != 0:
            loss /= step
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            step = 0
            loss = 0

        print(f"Epoch {epoch}, Loss: {sum_loss / len(dataloader)}")
        torch.save(
            model.state_dict(),
            f"D:\\AIChallenge\\Checkpoints\\model_epoch_{epoch}.pt",
        )
