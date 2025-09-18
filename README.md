
## Dataset Layout
Place videos like:
data/train/
cover_drive/.mp4
pull_shot/.mp4


## Setup
```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

Train

python src/train.py \
  --train_dir data/train \
  --test_dir data/test \
  --outputs_dir outputs \
  --epochs 40 --batch_size 8 --lr 1e-4 \
  --num_classes 5 --frame_h 100 --frame_w 100 --num_frames 14
