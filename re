#!/bin/zsh

usage="usage: $(basename "$0") [-h] [--gpu GPUs] name -- run an experiment

optional arguments:
  -h, --help\t show this help message and exit
  --gpu, -g\t a comma-separated list of GPU ids

positional arguments:
  name\t the name of the experiment
"

RUN_CMD="python3.6 main.py"
SRC_FILES=("py" "sh")

zparseopts -D -E -- g:=gpu -gpu:=gpu h=help -help=help
name=$1
gpus=${gpu[2]}

if [[ -n "$help" ]]; then
  echo -n "$usage" && exit
fi

if [[ -z "$name" ]]; then
  [[ -z "$gpus" ]] && echo -n "$usage" && exit
  [[ -n "$gpus" ]] && echo "error: a name must be specified" >&2 && exit 1
elif ! [[ "$name" =~ "^[A-Za-z][A-Za-z_0-9]*$" ]]; then
  echo "error: invalid name: $name" >&2; exit 1
fi
shift 1

if [[ -n "$gpus" ]]; then
  if ! [[ $gpus =~ "^[0-9](,[0-9])*$" ]]; then
    echo "error: invalid gpu list: $gpus" >&2; exit 1
  fi
  CVD="CUDA_VISIBLE_DEVICES=$gpus"
fi

# start at top of root worktree
cd $(git rev-parse --show-toplevel)
while ! [[ "$(git rev-parse --git-dir)" == ".git" ]]; do
  cd .. && cd $(git rev-parse --show-toplevel)
done

# check if experiment/branch already exists
git rev-parse --verify "$name" > /dev/null 2>&1
if [[ $? -eq 0 ]]; then
  read "?Experiment \"$name\" already exists. Recreate? [yN] " newp
  ! [[ $newp =~ "^[Yy]$" ]] && exit

  # clean up existing experiment
  rm -rf "experiments/$name"
  git worktree prune
  git branch -q -D "$name"
fi

# add any changed or untracked source files
for ext in $SRC_FILES; do
  git add -A "*.$ext" 2>/dev/null
done
git commit -q -m "setup experiment: $name" 2>/dev/null

# make a new worktree for the experiment
mkdir -p experiments
git worktree add "experiments/$name" -B "$name" -f > /dev/null 2>&1
treed=$?
git reset -q HEAD~1 2>/dev/null
if [[ $treed -ne 0 ]]; then
  echo "error: could not create worktree" && exit 1;
fi

# link in the data dir
ln -s $(pwd)/data "experiments/$name"

# run the experiment
cd "experiments/$name"
cmd="$CVD $RUN_CMD $@"
eval "$cmd"
