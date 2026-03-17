# Reusing Trajectories in Deep Policy Optimization Methods
Repository for the research project investigating whther the reuse of historical trajectories is beneficial for deep policy optimization methods.

## Setup and Installation

A `setup.sh` script is provided to automatically build the environment, ensuring compatibility across different operating systems.

### Prerequisites

Before you begin, ensure you have Anaconda or Miniconda installed on your system. If you are setting up a new Unix server, you can install Anaconda via command line:
`wget https://repo.anaconda.com/archive/Anaconda3-latest-Linux-x86_64.sh` and run it with `bash Anaconda3-latest-Linux-x86_64.sh`.

### Installation Steps

**1. Clone the repository**
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```

**2. Make the setup script executable**
Depending on your system's permissions, you may need to grant execution rights to the setup script:

```bash
chmod +x setup.sh
```

**3. Run the setup script**
Execute the script to build the environment:

```bash
./setup.sh
```

**4. Activate the environment**
Once the setup is complete, activate the Conda environment to start working:

```bash
conda activate rtpg
```