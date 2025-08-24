# デカ・オート (Deca-Auto)

**Deca-Auto: Automated Optimization of Decoupling Capacitor Combinations in PDN**

デカップリングコンデンサの組み合わせ最適解を自動探索するツールです。  
`user_config.toml` で定義されたコンデンサをもとに、目標マスクと比較して優れた合成インピーダンスを持つ組み合わせを探索します。

---

## 概要
- デカップリングコンデンサの自動探索 
- `SPICE`モデルを利用可能  
- コンデンサの基本パラメータ（C / ESR / ESL）を自分で定義して利用可能  

---

## 👤 Author
- YAE

## 📄 License
- MIT (お好きに役立ててください)

---

## 使い方

### 1. 設定方法
以下のいずれかでコンデンサを定義します：

- **SPICEモデルを利用する場合**  
  `.mod` ファイルのパスを `user_config.toml` に定義します。
  
- **パラメータを直接定義する場合**  
  `user_config.toml` 内で各コンデンサの C / ESR / ESL などのパラメータを定義します。

---

### 2. 実行方法
```bash
python -m deca_auto.main --toml user_config.toml
