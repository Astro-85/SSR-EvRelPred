# Clean all leftover files from specified run

RUN_ID=$1

for DIR in 'cfg_logs' 'model_epochs' 'predictions' 'tb_logs'; do
	if [ -d ./tmp/$DIR/$RUN_ID ]; then
		rm -rf ./tmp/$DIR/$RUN_ID
		echo Deleted ./tmp/$DIR/$RUN_ID
	fi
done

if [ -f ./tmp/ext_logs/$RUN_ID.txt ]; then 
	rm ./tmp/ext_logs/$RUN_ID.txt
	echo Deleted ./tmp/ext_logs/$RUN_ID.txt
fi

if [ -f ./tmp/models/$RUN_ID.pth ]; then
	rm ./tmp/models/$RUN_ID.pth
	echo Deleted ./tmp/models/$RUN_ID.pth
fi

if [ -f ./tmp/txt_logs/$RUN_ID.txt ]; then
	rm ./tmp/txt_logs/$RUN_ID.txt
	echo Deleted ./tmp/txt_logs/$RUN_ID.txt
fi

