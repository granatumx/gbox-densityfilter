FROM granatumx/gbox-py-sdk:1.0.0

RUN apt-get update
RUN apt install -y fontconfig
RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections
RUN apt install -y ttf-mscorefonts-installer
RUN fc-cache -f

COPY . .

RUN ./GBOXtranslateVERinYAMLS.sh
RUN tar zcvf /gbox.tgz package.yaml yamls/*.yaml
