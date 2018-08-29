all: client serverVideo serverVideo2

client:
	g++ -pthread client.cpp -o client `pkg-config opencv --cflags --libs`

serverVideo:
	g++ -pthread serverVideo.cpp -o serverVideo `pkg-config opencv --cflags --libs` 

serverVideo2:
	g++ -pthread serverVideo2.cpp -o serverVideo2 `pkg-config opencv --cflags --libs`

clean:
	rm -f client serverVideo serverVideo2
