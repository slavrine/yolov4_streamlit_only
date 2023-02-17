FROM mambaorg/micromamba:0.15.3
USER root
RUN mkdir /opt/drone_video_analysis
RUN chmod -R 777 /opt/drone_video_analysis
WORKDIR /opt/drone_video_analysis
USER micromamba
COPY environment.yml environment.yml
RUN micromamba install -y -n base -f environment.yml && \
   micromamba clean --all --yes
COPY run.sh run.sh
COPY project_contents project_contents
COPY nginx.conf /etc/nginx/nginx.conf
USER root
RUN chmod a+x run.sh
CMD ["./run.sh"]