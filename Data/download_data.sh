
#Downloading contiguous videos in zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Qn_QlDOPAAYj7CrehU8aVnng2CpvFlg5' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Qn_QlDOPAAYj7CrehU8aVnng2CpvFlg5" -O contiguous_videos.zip
#Unzipping the videos
unzip contiguous_videos.zip
#Removing the zip file
rm contiguous_videos.zip
#Downloading short gap videos in zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Rc2T_DUrzzJ7K2e5f_nH21_ZeVzwPaxa' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Rc2T_DUrzzJ7K2e5f_nH21_ZeVzwPaxa" -O short_gap.zip
#Unzipping the videos
unzip short_gap.zip
#Removing the zip file
rm short_gap.zip
#Downloading long gap videos in zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ZpU3YuZwwcZH033dxbqZm5jZLqczJdiu' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ZpU3YuZwwcZH033dxbqZm5jZLqczJdiu" -O long_gap.zip
#Unzipping the videos
unzip long_gap.zip
#Removing the zip file
rm long_gap.zip
#
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1OGDUypVKR_lqwTwIXNUkVU0VR5z4fymL' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1OGDUypVKR_lqwTwIXNUkVU0VR5z4fymL" -O Annotation.zip
