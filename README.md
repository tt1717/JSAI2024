# Atariゲームに対するTransformerベース強化学習のロバスト性検証

髙野剛志 ∗1, 計良宥志 ∗2, 川本一彦 ∗2

∗1千葉大学大学院融合理工学府, ∗2千葉大学大学院工学研究院

## プロジェクトページ

https://sites.google.com/view/jsai2024-japanese/%E3%83%9B%E3%83%BC%E3%83%A0

## 動作確認環境

- OS: Ubuntu 20.04, macOS
- プログラミング言語: Python 3.8
- ライブラリ・ツール:
  - PyTorch 1.9.0
  - gym 0.21.0
  - atari-py 0.2.9
  - numpy 1.21.0
  - pandas 1.3.0
  - matplotlib 3.4.0

## インストール方法

1. リポジトリをクローンします：
   ```bash
   git clone https://github.com/tt1717/JSAI2024.git
   cd JSAI2024
   ```

2. Conda環境を作成し、必要なライブラリをインストールします：
   ```bash
   conda env create -f conda_env.yml
   conda activate jsai2024
   ```

## 使用方法

1. データセットの作成：
   ```bash
   python create_dataset.py
   ```

2. 学習の実行：
   ```bash
   python run_dt_atari.py
   ```

3. 特定のゲームでの学習実行：
   ```bash
   ./run_dt_noiserate_seaquest.sh
   ```

## プロジェクト構造

- `run_dt_atari.py`: メインの学習スクリプト
- `create_dataset.py`: データセット作成スクリプト
- `fixed_replay_buffer.py`: リプレイバッファの実装
- `mingpt/`: GPTモデルの実装
- `conda_env.yml`: 環境設定ファイル

## ベースコード

このプロジェクトは以下のリポジトリをベースにしています：
https://github.com/kzl/decision-transformer

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。
