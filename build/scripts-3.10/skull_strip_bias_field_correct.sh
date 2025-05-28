#!/bin/sh
#   Copyright (C) 2021 University of Oxford
#   SHCOPYRIGHT
#set -e
#set -x

######

if [ $# -lt 2 ] ; then
  echo "Usage: `basename $0` <input_image_name> <output_basename>"
  echo " "
  echo "The script applies the preprocessing pipeline on the input image to be used in microbleednet with a specified output basename"
  echo "input_image_name = 	name of the input unprocessed FLAIR image"
  echo "output_basename = 	name to be used for the processed FLAIR and T1 images (along with the absolute path); output_basename_FLAIR.nii.gz, output_basename_T1.nii.gz and output_basename_WMmask.nii.gz will be saved"
  exit 0
fi

inpfile=$1
# echo $inpfile
inpimg=`basename ${inpfile} .nii.gz`
inpdir=`dirname ${inpfile} `
pushd $inpdir > /dev/null
inpdir=`pwd`
popd > /dev/null

outbasename=$2
# echo $outbasename
outname=`basename ${outbasename}`
outdir=`dirname ${outbasename}`
pushd $outdir > /dev/null
outdir=`pwd`
popd > /dev/null

# SPECIFY ORIGINAL DIRECTORY
origdir=`pwd`

# CREATE TEMPORARY DIRECTORY
logID=`echo $(date | awk '{print $1 $2}' |  sed 's/://g')`
TMPVISDIR=`mktemp -d ${outdir}/truenet_${logID}_${inoimg}_XXXXXX`

# REORIENTING FLAIR AND T1 IMAGES TO STD SPACE
$FSLDIR/bin/fslreorient2std ${inpfile}.nii.gz ${TMPVISDIR}/INPUT.nii.gz

# PREPROCESSING OF FLAIR IMAGE
$FSLDIR/bin/bet ${TMPVISDIR}/INPUT.nii.gz ${TMPVISDIR}/INPUT_brain.nii.gz
$FSLDIR/bin/fast -B --nopve ${TMPVISDIR}/INPUT_brain.nii.gz
${FSLDIR}/bin/imcp ${TMPVISDIR}/INPUT_brain_restore.nii.gz ${outdir}/${outname}_preproc.nii.gz

# REMOVES TEMPORARY DIRECTORY
rm -r ${TMPVISDIR}

exit 0