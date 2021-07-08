
#Downloading contiguous videos in zip
wget --load-cookies cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1tp8Y576PTFPiduE0_hu6fAPp4iFWG53Z' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1tp8Y576PTFPiduE0_hu6fAPp4iFWG53Z" -O contiguous_videos.zip
#Unzipping the videos
unzip contiguous_videos.zip
#Removing the zip file
rm contiguous_videos.zip
#Downloading short gap videos in zip
wget --load-cookies cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1vpVvL43v93U0Cd8Kpv6kpUQSEgKoNp1c' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1vpVvL43v93U0Cd8Kpv6kpUQSEgKoNp1c" -O short_gap.zip
#Unzipping the videos
unzip short_gap.zip
#Removing the zip file
rm short_gap.zip
#Downloading long gap videos in zip
wget --load-cookies cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1L_C8iUWzIPqmJuci4RRFqct9yUpOrb82' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1L_C8iUWzIPqmJuci4RRFqct9yUpOrb82" -O long_gap.zip
#Unzipping the videos
unzip long_gap.zip
#Removing the zip file
rm long_gap.zip
#
wget --load-cookies cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1OGDUypVKR_lqwTwIXNUkVU0VR5z4fymL' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1OGDUypVKR_lqwTwIXNUkVU0VR5z4fymL" -O Annotation.json
