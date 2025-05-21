# Atariゲームに対するTransformerベース強化学習のロバスト性検証
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 1.9.0](https://img.shields.io/badge/PyTorch-1.9.0-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![English](https://img.shields.io/badge/English-README.md-red)](README.md)

## 🌟 概要
本研究では、Transformerアーキテクチャをベースとした強化学習手法を用いて、Atariゲームにおけるロバスト性の検証を行います。特に、ノイズや外乱に対する耐性を評価し、より安定した学習と推論を実現することを目指しています。

## ✨ デモ
![デモ](images/demo.gif)

## 🚀 特徴
- Transformerベースの強化学習モデルの実装
- 複数のAtariゲームでのロバスト性評価
- ノイズ耐性の定量的評価
- 効率的なデータ収集と学習パイプライン

## 🛠️ インストール
```bash
# リポジトリのクローン
git clone https://github.com/tt1717/JSAI2024.git
cd JSAI2024

# Conda環境の作成とライブラリのインストール
conda env create -f conda_env.yml
conda activate jsai2024
```

## 🔥 クイックスタート
```bash
# データセットの作成
python create_dataset.py

# 学習の実行
python run_dt_atari.py

# 特定のゲームでの学習実行
./run_dt_noiserate_seaquest.sh
```

## 🏗️ プロジェクト構造
```
JSAI2024/
├── run_dt_atari.py          # メインの学習スクリプト
├── create_dataset.py        # データセット作成スクリプト
├── fixed_replay_buffer.py   # リプレイバッファの実装
├── mingpt/                  # GPTモデルの実装
├── materials/              # 実験用データ
├── images/                 # 画像・GIFファイル
└── conda_env.yml          # 環境設定ファイル
```

## 📈 学習と評価
- 学習環境: Ubuntu 20.04, macOS
- 使用ライブラリ:
  - PyTorch 1.9.0
  - gym 0.21.0
  - atari-py 0.2.9
  - numpy 1.21.0
  - pandas 1.3.0
  - matplotlib 3.4.0

## 📝 使用例
```python
# モデルの学習
python run_dt_atari.py --game seaquest --noise_rate 0.1

# 評価の実行
python run_dt_atari.py --mode eval --game seaquest
```

## 📚 データセット
- Atari Learning Environment (ALE)のゲームデータを使用
- 各ゲームのリプレイデータを自動収集
- ノイズ付加によるロバスト性評価用データセット

## 🤝 貢献方法
1. このリポジトリをフォーク
2. 新しいブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 🪪 ライセンス
このプロジェクトはMITライセンスの下で公開されています。
詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 📝 引用
```bibtex
@article{jsai2024,
  title={Atariゲームに対するTransformerベース強化学習のロバスト性検証},
  author={髙野剛志 and 計良宥志 and 川本一彦},
  journal={人工知能学会全国大会論文集},
  year={2024}
}
```

## 🙏 謝辞
このプロジェクトは以下のリポジトリをベースにしています：
- [Decision Transformer](https://github.com/kzl/decision-transformer) 