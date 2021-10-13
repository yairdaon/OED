rsync -av -e ssh --exclude-from 'exclude_list.txt' . monster:OED
ssh monster "killer; cd OED; rm nohup.out; nohup python3 main.py &" 
