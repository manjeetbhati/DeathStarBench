for file in {14..150}
do	
         n=$(($file%13))
	echo $n
	str="${file}.png"
	inp="${n}.png"
	cp $inp $str
done
