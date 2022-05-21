from a2j.a2j import A2JModelLightning, A2JDataModule
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == "__main__":
    cli = LightningCLI(A2JModelLightning, A2JDataModule,  parser_kwargs={"fit": {"default_config_files": ["config/a2j.yaml"]}, "test": {"default_config_files": ["config/a2j.yaml"]} }, trainer_defaults={"logger": WandbLogger(project="HandNet-Pipeline", save_dir='wandb/a2j', name="A2J"), "callbacks": ModelCheckpoint(monitor="val_accuracy")}, save_config_overwrite=True)