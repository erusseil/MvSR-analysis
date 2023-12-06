
listcommon="polynomial0 polynomial_partial friedman2"
listsin="friedman1"

for i in $listcommon; do
    nohup python analysis.py --function $i --maxL 5 7 9 11 13 &
    nohup python analysis.py --function $i --maxL 15 17 19 21 &
    nohup python analysis.py --function $i --maxL 23 25 &

done

for i in $listsin; do
    nohup python analysis.py --function $i --maxL 5 7 9 11 13 --opset sin &
    nohup python analysis.py --function $i --maxL 15 17 19 21 --opset sin &
    nohup python analysis.py --function $i --maxL 23 25 --opset sin &

done
