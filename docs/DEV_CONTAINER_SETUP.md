# Dev Container Setup

Dev Container開発環境の構築手順。

## 前提条件

- VS Code
- Docker

インストールされていない場合は[インストールガイド](./docs/INSTALL.md)を参照。

## 手順

1. VS Codeを開き、`Ctrl+Shift+X`キーを押して拡張機能パネルを開く。

2. **RECOMMENDED**セクションから「Remote - Containers」を選択し、インストールする。

3. コマンドパレット (`Ctrl+Shift+P`) を開き、`Remote-Containers: Reopen in Container`を選択する。

4. コンテナがビルドされ、必要な依存パッケージやツールが自動的が自動的にインストールされる。この作業には数分かかる場合がある。

5. コマンドパレット (`Ctrl+Shift+P`) で`Dev Containers: Show Container Log`を選択し、ビルドログを確認する。インストールに失敗した拡張機能があれば手動でインストールする。