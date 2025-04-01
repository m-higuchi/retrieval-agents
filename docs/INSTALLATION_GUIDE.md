# Installation Guide

各種ツール等のインストール手順。

## Windows

### VS Code

[公式サイト](https://code.visualstudio.com/download)からWindows版インストーラをダウンロードする。ダウンロードされたインストーラ(`.exe`)を実行し、セットアップウィザードに従ってインストールする。


### WSL2

管理者権限でPowershellまたはコマンドプロンプトを開き、`wsl --install`コマンドを実行し、再起動する。

```
wsl --install
```
これにより規定のLinuxディストリビューションとしてUbuntuがインストールされる。
再起動後、スタートメニューからUbuntuを開くと、初回起動時にUbuntuのユーザー名とパスワードの作成を求められるので作成する。


> [!WARNING]
> Windows 10 バージョン 2004 以上 (ビルド 19041 以上) または Windows 11より前のバージョンの場合は[以前のバージョンの WSL の手動インストール手順](https://learn.microsoft.com/ja-jp/windows/wsl/install-manual)に従ってインストールする。




### Docker Desktop for Windows

[公式サイト](https://www.docker.com/ja-jp/get-started/)からWindows版インストーラ(AMD64)をダウンロードし、ダウンロードされたインストーラ(`.exe`)を実行する。インストールが完了すると再起動かサインアウトを要求されるため指示に従う。その後、自動的にDocker Desktopが起動するので指示に従ってセットアップする。オプションは基本的にデフォルトで問題ない。

### Git

[公式サイト](https://git-scm.com/downloads/win)から最新バージョンのインストーラをダウンロードし、ダウンロードされたインストーラ(`.exe`)を実行する。インストールオプションは以下の項目以外はデフォルトで問題ない。

- **Choose the default editor used by Git:** `Use Visual Studio Code as Git’s default editor`（推奨）
- **Adjusting your PATH environment:** `Git from the command line and also from 3rd-party softwar`（必須）
- **Configuring the line ending conversions:** `Checkout Windows-style, commit Unix-style line endings`（必須）

インストール完了後、コマンドプロンプトでgitのユーザー名とメールアドレスを設定する。

```
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

> [!NOTE]
> 基本的にはDev Containerで開発を行うことを想定しているが、ホストOS上のGitはリポジトリのcloneで必要になる。

> [!NOTE]
> Dev Containerを使用する場合、Gitのグローバル設定はDev Container内に引き継がれないため上記の設定はリポジトリ設定(`--global`オプションを付けない)で行う。