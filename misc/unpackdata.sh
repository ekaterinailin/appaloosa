# unpack compressed files

cd ..
cat Q16_public/*.tgz | tar -izx -C Q16_public
cat Q17_public/*.tgz | tar -izx -C Q17_public

# remove compressed files to save disk space
rm Q*_public/*.tgz
