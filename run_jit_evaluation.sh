projects=( eclipse freedesktop gcc kde linux openoffice )
for project in ${projects[@]}
do
    jit_types=( after_jit before_jit )
    for jit_type in ${jit_types[@]}
    do
	for ((i=1;i<=5;i++))
        do
            echo $project
            echo $jit_type
            echo $i
            python3 evaluation_before_after_jit.py $project $jit_type $i
	done
    done
done
