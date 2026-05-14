# MagicRL3

## Setup and Installation

A `setup.sh` script is provided to automatically build the environment, ensuring compatibility across different operating systems.

### Prerequisites

Before you begin, ensure you have Anaconda or Miniconda installed on your system. If you are setting up a new Unix server, you can install Anaconda via command line:
`wget https://repo.anaconda.com/archive/Anaconda3-latest-Linux-x86_64.sh` and run it with `bash Anaconda3-latest-Linux-x86_64.sh`.

### Installation Steps

**1. Clone the repository**
```bash
git clone https://github.com/MontenegroAlessandro/RT-DeepRL.git [your_repo_name]
```

or 
```bash
git clone git@github.com:MontenegroAlessandro/MagicRL3.git [your_repo_name]
```

then 
```bash
cd your_repo_name
```

**Note**: if you do not specify `your_repo_name`, the default folder name will be `MagicRL3`.

**2. Make the setup script executable**
Depending on your system's permissions, you may need to grant execution rights to the setup script:

```bash
chmod +x setup.sh
```

**3. Run the setup script**
Execute the script to build the environment:

```bash
./setup.sh [personal_env_name]
```

**4. Activate the environment**
Once the setup is complete, activate the Conda environment to start working:

```bash
conda activate personal_env_name
```

**Note**: if you do not specify `personal_env_name`, then the default one will be `rtpg`.