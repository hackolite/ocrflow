FROM alpine:3.19.0
ENV FTP_USER=$FTP_USER \
	FTP_PASS=$FTP_PASSWORD \
	GID=1000 \
	UID=1000


RUN apk add --no-cache bash
RUN apk add --no-cache --update vsftpd==3.0.5-r2

COPY [ "./vsftpd.conf", "/etc" ]
COPY [ "./docker-entrypoint.sh", "." ]

# Créez un répertoire pour stocker les données du FTP
RUN mkdir -p ./data
# Ajoutez une instruction pour définir le point de montage dans le conteneur
VOLUME ./data

ENTRYPOINT [ "./docker-entrypoint.sh" ]
EXPOSE 20/tcp 21/tcp 40000-40009/tcp
HEALTHCHECK CMD netstat -lnt | grep :21 || exit 1
