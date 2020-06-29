N=`ls .*.files`
echo Makefile > $N
echo TODO.md >> $N
echo update_files.sh >> $N
# echo TODO.txt >> $N
for d in src tests examples
do
    for t in '*.h' '*.hpp' '*.cpp' '*.tcc' '*.cu' '*.txt' '*.py' '*.js' '*.html' '*.css' '*.files' '*.met' '*.coffee' '*.asm'
    do
        for i in `find $d -name "$t" -a -not -wholename "*/compilations/*"`
        do
            echo $i >> $N
            echo $i
            # git add $i
        done
    done
done
