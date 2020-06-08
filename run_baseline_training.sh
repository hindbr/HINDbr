projects=( eclipse freedesktop gcc gnome kde libreoffice linux llvm openoffice )
for project in ${projects[@]}
do
    python3 baseline.py $project 
done
