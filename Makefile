include MODEL.mk

# Update BASE to ImageNet directory
# Update WANDB_ACCOUNT if you want to visualize the training curves
BASE :=
WANDB_ACCOUNT :=

# ************************************ VENV ************************************ #
SHELL := /bin/bash
VENV_NAME := ra-principles
CONDA_ACTIVATE := source $$(conda info --base)/etc/profile.d/conda.sh && conda deactivate && conda activate $(VENV_NAME)
PYTHON := $(CONDA_ACTIVATE) && python
PIP := $(CONDA_ACTIVATE) && pip

# Install advertorch from github due to its usage of obsoleted code from pytorch
# https://github.com/BorealisAI/advertorch/issues/99
clean:
	rm -rf lib/*
	rm -f .venv_done

.venv_done: clean
	conda create -n $(VENV_NAME) python=3.9
	$(PIP) install -r requirements.txt
	cd lib && git clone https://github.com/NVIDIA/apex
	$(PIP) install -v --disable-pip-version-check --no-cache-dir --no-build-isolation lib/apex
	cd lib && git clone https://github.com/BorealisAI/advertorch.git
	$(PIP) install -e lib/advertorch
	$(PIP) install -e .
	touch $@

# ************************ Adversarial Training Configs ************************ #
IN_DIR := $(wildcard $(BASE)/*imagenet*)
check_dir:
ifneq ("$(IN_DIR)","")
	$(info ImageNet folder detected at: $(IN_DIR))
else
	$(error Please set BASE in Makefile to the directory that stores ImageNet)
endif
ifneq ("$(WANDB_ACCOUNT)","")
	$(info Training curves are saved at $(WANDB_ACCOUNT)/Ra-Principles under your WandB projects.)
else
	$(info If you want to visualize the training curves, please set WANDB_ACCOUNT in Makefile.)
endif

SRC = robustarch
assign-vars = $(foreach A,$2,$(eval $1: $A))

# FAT - 3 Phases ImageNet
PHASE1 = cd $(SRC) && $(PYTHON) -m main train_test=fat_phase1 ++name=$(NAME) ++visualization.entity=$(WANDB_ACCOUNT) ++dataset.data_dir="$(BASE)/imagenet-sz/160"
PHASE2 = cd $(SRC) && $(PYTHON) -m main train_test=fat_phase2 ++name=$(NAME) ++visualization.entity=$(WANDB_ACCOUNT) ++dataset.data_dir="$(BASE)/imagenet-sz/352"
PHASE3 = cd $(SRC) && $(PYTHON) -m main train_test=fat_phase3 ++name=$(NAME) ++visualization.entity=$(WANDB_ACCOUNT) ++dataset.data_dir="$(BASE)/imagenet"
TEST = cd $(SRC) && $(PYTHON) -m main train_test=test ++name=$(NAME) ++visualization.entity=$(WANDB_ACCOUNT)
TRAIN_EPS4 = ++attack.train.eps=5 ++attack.test.eps=4

# Standard PGD AT - ImageNet
TRAIN_PGD = cd $(SRC) && $(PYTHON) -m main train_test=at_pgd attack=train ++name=$(NAME) ++visualization.entity=$(WANDB_ACCOUNT) ++dataset.data_dir="$(BASE)/imagenet"
TEST_PGD = cd $(SRC) && $(PYTHON) -m main train_test=at_pgd_test attack=test ++name=$(NAME) ++visualization.entity=$(WANDB_ACCOUNT) ++dataset.data_dir="$(BASE)/imagenet"
TRAIN_PGD3_EPS4 = ++attack.train.eps=4 ++attack.train.gamma=2.67 ++attack.test.eps=4 ++attack.train.step=3
STEP_LR1 = ++train_test.schedule="step" ++train_test.decay_t=30 ++train_test.decay_rate=0.1 ++train_test.warmup_epochs=10 ++train_test.warmup_lr=0.1 ++train_test.lr=0.1
COLOR_JITTER = ++train_test.color_jitter=0.1
LIGHTING = ++train_test.lighting=true
TRAIN_CONFIG1 = $(TRAIN_PGD3_EPS4) $(STEP_LR1) $(COLOR_JITTER) $(LIGHTING) ++attack.train.eps_schedule=null ++train_test.cooldown_epochs=0

# *********************************** Attacks ********************************** #
# PGD
TEST_PGD10_2-1 = ++attack.test.eps=2 ++attack.test.gamma=1 ++attack.test.step=10
TEST_PGD10_4-1 = ++attack.test.eps=4 ++attack.test.gamma=1 ++attack.test.step=10
TEST_PGD10_8-1 = ++attack.test.eps=8 ++attack.test.gamma=2 ++attack.test.step=10
TEST_PGD50_4-1 = ++attack.test.eps=4 ++attack.test.gamma=1 ++attack.test.step=50
TEST_PGD100_2-1 = ++attack.test.eps=2 ++attack.test.gamma=1 ++attack.test.step=100
TEST_PGD100_4-1 = ++attack.test.eps=4 ++attack.test.gamma=1 ++attack.test.step=100
TEST_PGD100_8-1 = ++attack.test.eps=8 ++attack.test.gamma=1 ++attack.test.step=100

# AutoAttack
AUTOATTACK_IN = ++attack.test.name=aa ++attack.test.batch_size=256 ++attack.test.n_examples=5000 ++attack.test.eps=4


# ******************************** EXPERIMENTS ********************************* #
# ************************************ FAT ************************************* #
# ResNet-50
NAME = Torch_ResNet50
$(call assign-vars, experiments/$(NAME)/.done_train, NAME=Torch_ResNet50 ARCH=$(RESNET50_TORCH))
experiments/$(NAME)/.done_train:
	$(PHASE1) $(TORCHMODEL) $(TRAIN_EPS4)
	$(PHASE2) $(TORCHMODEL) $(TRAIN_EPS4)
	$(PHASE3) $(TORCHMODEL) $(TRAIN_EPS4)
	touch $@

$(call assign-vars, experiments/$(NAME)/.done_test_pgd, NAME=Torch_ResNet50 ARCH=$(RESNET50_TORCH))
experiments/$(NAME)/.done_test_pgd: experiments/$(NAME)/.done_train
	$(TEST) $(TORCHMODEL) $(TRAIN_EPS4) $(TEST_PGD10_2-1)
	$(TEST) $(TORCHMODEL) $(TRAIN_EPS4) $(TEST_PGD10_4-1)
	$(TEST) $(TORCHMODEL) $(TRAIN_EPS4) $(TEST_PGD10_8-1)
	touch $@

# Ra ResNet-50
NAME = RaResNet50
$(call assign-vars, experiments/$(NAME)/.done_train, NAME=RaResNet50)
experiments/$(NAME)/.done_train:
	$(PHASE1) $(TRAIN_EPS4) $(RESNET50) $(RaResNet50)
	$(PHASE2) $(TRAIN_EPS4) $(RESNET50) $(RaResNet50)
	$(PHASE3) $(TRAIN_EPS4) $(RESNET50) $(RaResNet50)
	touch $@

$(call assign-vars, experiments/$(NAME)/.done_test_pgd, NAME=RaResNet50)
experiments/$(NAME)/.done_test_pgd: experiments/$(NAME)/.done_train
	$(TEST) $(TRAIN_EPS4) $(TEST_PGD10_2-1) $(RESNET50) $(RaResNet50)
	$(TEST) $(TRAIN_EPS4) $(TEST_PGD10_4-1) $(RESNET50) $(RaResNet50)
	$(TEST) $(TRAIN_EPS4) $(TEST_PGD10_8-1) $(RESNET50) $(RaResNet50)
	touch $@

# Ra ResNet-101
NAME = RaResNet101
$(call assign-vars, experiments/$(NAME)/.done_train, NAME=RaResNet101)
experiments/$(NAME)/.done_train:
	$(PHASE1) $(TRAIN_EPS4) $(RESNET50) $(RaResNet101)
	$(PHASE2) $(TRAIN_EPS4) $(RESNET50) $(RaResNet101)
	$(PHASE3) $(TRAIN_EPS4) $(RESNET50) $(RaResNet101)
	touch $@

$(call assign-vars, experiments/$(NAME)/.done_test_pgd, NAME=RaResNet101)
experiments/$(NAME)/.done_test_pgd: experiments/$(NAME)/.done_train
	$(TEST) $(TRAIN_EPS4) $(TEST_PGD10_2-1) $(RESNET50) $(RaResNet101)
	$(TEST) $(TRAIN_EPS4) $(TEST_PGD10_4-1) $(RESNET50) $(RaResNet101)
	$(TEST) $(TRAIN_EPS4) $(TEST_PGD10_8-1) $(RESNET50) $(RaResNet101)
	touch $@

# Ra WRN-101-2
NAME = RaWRN101_2
$(call assign-vars, experiments/$(NAME)/.done_train, NAME=RaWRN101_2)
experiments/$(NAME)/.done_train:
	$(PHASE1) $(TRAIN_EPS4) $(RESNET50) $(RaWRN101_2) ++train_test.lr_values.1=0.25
	$(PHASE2) $(TRAIN_EPS4) $(RESNET50) $(RaWRN101_2)
	$(PHASE3) $(TRAIN_EPS4) $(RESNET50) $(RaWRN101_2)
	touch $@

$(call assign-vars, experiments/$(NAME)/.done_test_pgd, NAME=RaWRN101_2)
experiments/$(NAME)/.done_test_pgd: experiments/$(NAME)/.done_train
	$(TEST) $(TRAIN_EPS4) $(TEST_PGD10_2-1) $(RESNET50) $(RaWRN101_2)
	$(TEST) $(TRAIN_EPS4) $(TEST_PGD10_4-1) $(RESNET50) $(RaWRN101_2)
	$(TEST) $(TRAIN_EPS4) $(TEST_PGD10_8-1) $(RESNET50) $(RaWRN101_2)
	touch $@

# ****************************** Standard PGD AT ******************************* #
# Ra ResNet-50
NAME = PGDAT_RaResNet50
$(call assign-vars, experiments/$(NAME)/.done_train, NAME=PGDAT_RaResNet50)
experiments/$(NAME)/.done_train:
	$(TRAIN_PGD) $(TRAIN_CONFIG1) $(RESNET50) $(RaResNet50)
	touch $@

$(call assign-vars, experiments/$(NAME)/.done_test_pgd, NAME=PGDAT_RaResNet50)
experiments/$(NAME)/.done_test_pgd: experiments/$(NAME)/.done_train
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD10_4-1) $(RESNET50) $(RaResNet50)
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD50_4-1) $(RESNET50) $(RaResNet50)
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD100_4-1) $(RESNET50) $(RaResNet50)
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD100_8-1) $(RESNET50) $(RaResNet50)
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD100_2-1) $(RESNET50) $(RaResNet50)
	touch $@

$(call assign-vars, experiments/$(NAME)/.done_test_aa, NAME=PGDAT_RaResNet50)
experiments/$(NAME)/.done_test_aa: experiments/$(NAME)/.done_train
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(AUTOATTACK_IN) $(RESNET50) $(RaResNet50)
	touch $@

RARESNET50_WEIGHTS = trained_models/pretrained/ra_resnet50_imagenet.pt
$(call assign-vars, experiments/$(NAME)/.done_test_pretrained, NAME=PGDAT_RaResNet50)
experiments/$(NAME)/.done_test_pretrained:
ifneq ("$(wildcard $(RARESNET50_WEIGHTS))", "")
	$(info RaResNet50 weights file already exists.)
else
	wget -c https://huggingface.co/poloclub/robust-principles/resolve/main/ra_resnet50_imagenet.pt -P trained_models/pretrained
	$(info RaResNet50 weights file are successfully downloaded.)
endif
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(AUTOATTACK_IN) $(RESNET50) $(RaResNet50) ++train_test.resume="../$(RARESNET50_WEIGHTS)"
	touch $@

# Ra ResNet-101
NAME = PGDAT_RaResNet101
$(call assign-vars, experiments/$(NAME)/.done_train, NAME=PGDAT_RaResNet101)
experiments/$(NAME)/.done_train:
	$(TRAIN_PGD) $(TRAIN_CONFIG1) $(RESNET50) $(RaResNet101)
	touch $@

$(call assign-vars, experiments/$(NAME)/.done_test_pgd, NAME=PGDAT_RaResNet101)
experiments/$(NAME)/.done_test_pgd: experiments/$(NAME)/.done_train
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD100_8-1) $(RESNET50) $(RaResNet101)
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD100_2-1) $(RESNET50) $(RaResNet101)
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD10_4-1) $(RESNET50) $(RaResNet101)
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD50_4-1) $(RESNET50) $(RaResNet101)
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD100_4-1) $(RESNET50) $(RaResNet101)
	touch $@

$(call assign-vars, experiments/$(NAME)/.done_test_aa, NAME=PGDAT_RaResNet101)
experiments/$(NAME)/.done_test_aa: experiments/$(NAME)/.done_train
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(AUTOATTACK_IN) $(RESNET50) $(RaResNet101)
	touch $@

RARESNET101_WEIGHTS = trained_models/pretrained/ra_resnet101_imagenet.pt
$(call assign-vars, experiments/$(NAME)/.done_test_pretrained, NAME=PGDAT_RaResNet101)
experiments/$(NAME)/.done_test_pretrained:
ifneq ("$(wildcard $(RARESNET101_WEIGHTS))", "")
	$(info RaResNet101 weights file already exists.)
else
	wget -c https://huggingface.co/poloclub/robust-principles/resolve/main/ra_resnet101_imagenet.pt -P trained_models/pretrained
	$(info RaResNet101 weights file are successfully downloaded.)
endif
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(AUTOATTACK_IN) $(RESNET50) $(RaResNet101) ++train_test.resume="../$(RARESNET101_WEIGHTS)"
	touch $@

# Ra WRN-101_2
NAME = PGDAT_RaWRN101_2
$(call assign-vars, experiments/$(NAME)/.done_train, NAME=PGDAT_RaWRN101_2)
experiments/$(NAME)/.done_train:
	$(TRAIN_PGD) $(TRAIN_CONFIG1) $(RESNET50) $(RaWRN101_2)
	touch $@

$(call assign-vars, experiments/$(NAME)/.done_test_pgd, NAME=PGDAT_RaWRN101_2)
experiments/$(NAME)/.done_test_pgd: experiments/$(NAME)/.done_train
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD10_4-1) $(RESNET50) $(RaWRN101_2)
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD50_4-1) $(RESNET50) $(RaWRN101_2)
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD100_4-1) $(RESNET50) $(RaWRN101_2)
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD100_8-1) $(RESNET50) $(RaWRN101_2)
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(TEST_PGD100_2-1) $(RESNET50) $(RaWRN101_2)
	touch $@

$(call assign-vars, experiments/$(NAME)/.done_test_aa, NAME=PGDAT_RaWRN101_2)
experiments/$(NAME)/.done_test_aa: experiments/$(NAME)/.done_train
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(AUTOATTACK_IN) $(RESNET50) $(RaWRN101_2)
	touch $@

RAWRN101_2_WEIGHTS = trained_models/pretrained/ra_wrn101_2_imagenet.pt
$(call assign-vars, experiments/$(NAME)/.done_test_pretrained, NAME=PGDAT_RaWRN101_2)
experiments/$(NAME)/.done_test_pretrained:
ifneq ("$(wildcard $(RAWRN101_2_WEIGHTS))", "")
	$(info RaWRN101_2 weights file already exists.)
else
	wget -c https://huggingface.co/poloclub/robust-principles/resolve/main/ra_wrn101_2_imagenet.pt -P trained_models/pretrained
	$(info RaWRN101_2 weights file are successfully downloaded.)
endif
	$(TEST_PGD) $(TRAIN_PGD3_EPS4) $(AUTOATTACK_IN) $(RESNET50) $(RaWRN101_2) ++train_test.resume="../$(RAWRN101_2_WEIGHTS)"
	touch $@

# *************************** CIFAR-10 Evaluation ****************************** #
AUTOATTACK_CIFAR = ++attack.test.name=aa ++attack.test.batch_size=256 ++attack.test.n_examples=10000 ++attack.test.eps=8
TEST_AA_CIFAR = cd $(SRC) && $(PYTHON) -m main dataset=cifar10 train_test=cifar_dm_test attack=test ++name=$(NAME) ++visualization.entity=$(WANDB_ACCOUNT)

# Ra WRN-70-16
NAME = DM50-RA-WRN-70-16-CIFAR10
RA_WRN70_16_WEIGHTS = trained_models/pretrained/ra_wrn70_16_cifar10.pt
$(call assign-vars, experiments/$(NAME)/.done_test_pretrained, NAME=DM50-RA-WRN-70-16-CIFAR10)
experiments/$(NAME)/.done_test_pretrained:
ifneq ("$(wildcard $(RA_WRN70_16_WEIGHTS))", "")
	$(info RaWRN70_16 weights file already exists.)
else
	wget -c https://huggingface.co/poloclub/robust-principles/resolve/main/ra_wrn70_16_cifar10.pt -P trained_models/pretrained
	$(info RaWRN70_16 weights file are successfully downloaded.)
endif
	$(TEST_AA_CIFAR) $(AUTOATTACK_CIFAR) $(RaWRN70_16) ++train_test.resume="../$(RA_WRN70_16_WEIGHTS)"
	touch $@

# ****************************************************************************** #