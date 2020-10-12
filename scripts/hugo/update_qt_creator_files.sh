N=`ls -a .*.files`
echo Makefile > $N
echo TODO.md >> $N
# echo TODO.txt >> $N
echo src/ppwl.cpp >> $N
for d in lib src tests scripts
do
    for t in '*.h' '*.tcc' '*.cpp' '*.cu' '*.txt' '*.py' '*.js' '*.html' '*.css' '*.files' '*.met' '*.coffee' '*.asm' '*.inc'
    do
        for i in `find $d -name "$t" -a -not -wholename "*/compilations/*"`
        do
            echo $i >> $N
            echo $i
            # git add $i
        done
    done
done
