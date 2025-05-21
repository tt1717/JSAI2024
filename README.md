# Robustness Verification of Transformer-based Reinforcement Learning for Atari Games
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 1.9.0](https://img.shields.io/badge/PyTorch-1.9.0-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![日本語](https://img.shields.io/badge/日本語-README_ja.md-blue)](README_ja.md)

## 🌟 Overview
This research implements a Transformer-based reinforcement learning approach to verify robustness in Atari games. We particularly focus on evaluating the model's resilience against noise and disturbances, aiming to achieve more stable learning and inference.

## ✨ Demo

![Demo](images/demo.gif)

## 🚀 Features
- Implementation of Transformer-based reinforcement learning model
- Robustness evaluation across multiple Atari games
- Quantitative assessment of noise tolerance
- Efficient data collection and learning pipeline

## 🛠️ Installation
```bash
# Clone the repository
git clone https://github.com/tt1717/JSAI2024.git
cd JSAI2024

# Create conda environment and install dependencies
conda env create -f conda_env.yml
conda activate jsai2024
```

## 🔥 Quick Start
```bash
# Create dataset
python create_dataset.py

# Run training
python run_dt_atari.py

# Run training for specific game
./run_dt_noiserate_seaquest.sh
```

## 🏗️ Project Structure
```
JSAI2024/
├── run_dt_atari.py          # Main training script
├── create_dataset.py        # Dataset creation script
├── fixed_replay_buffer.py   # Replay buffer implementation
├── mingpt/                  # GPT model implementation
├── materials/              # Experimental data
├── images/                 # Images and GIF files
└── conda_env.yml          # Environment configuration file
```

## 📈 Training & Evaluation
- Environment: Ubuntu 20.04, macOS
- Libraries:
  - PyTorch 1.9.0
  - gym 0.21.0
  - atari-py 0.2.9
  - numpy 1.21.0
  - pandas 1.3.0
  - matplotlib 3.4.0

## 📝 Usage Examples
```python
# Model training
python run_dt_atari.py --game seaquest --noise_rate 0.1

# Evaluation
python run_dt_atari.py --mode eval --game seaquest
```

## 📚 Dataset
- Uses Atari Learning Environment (ALE) game data
- Automatic collection of replay data for each game
- Dataset for robustness evaluation with added noise

## 🤝 Contributing
1. Fork this repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🪪 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📝 Citation
```bibtex
@article{jsai2024,
  title={Transformer-based Reinforcement Learning for Robust Atari Game Playing},
  author={Takano, Tsuyoshi and Kera, Hiroshi and Kawamoto, Kazuhiko},
  journal={Proceedings of the Annual Conference of the Japanese Society for Artificial Intelligence},
  year={2024}
}
```

## 🙏 Acknowledgements
This project is based on the following repository:
- [Decision Transformer](https://github.com/kzl/decision-transformer)
