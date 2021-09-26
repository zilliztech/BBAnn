pip3 install numpy sklearn faiss-cpu
sudo apt-get install openmpi-bin openmpi-common openssh-client openssh-server libopenmpi2 libopenmpi-dev openmpi-doc

echo "Installing kahip at /tmp: https://github.com/KaHIP/KaHIP"
cd /tmp
git clone https://github.com/KaHIP/KaHIP
sudo ./KaHIP/compile_withcmake.sh
cd -
echo "kahip installed at /tmp/KaHIP"

