wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1taRfjpRLN9LlTSkQSZ75F-k3WXJRxdJT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1taRfjpRLN9LlTSkQSZ75F-k3WXJRxdJT" -O '../data/train_data.zip' && rm -rf /tmp/cookies.txt

unzip ../data/train_data.zip -d ../data

rm -rf ../data/train_data.zip

mv ../data/drive/My\ Drive/Pose_generation/* ../data