rsync -av -e ssh --exclude-from 'exclude_list.txt' . monster:OED
# ssh monster:OED #  "cd OED rm nohup.out; nohup python3 main.py &"
ssh -t monster 'cd OED && exec bash -l'