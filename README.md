# Addressing Bias and Representativeness in Digital Heritage Collections

Thsi repository contains data and code for the paper:

Kaspar Beelen, Jon Lawrence, Daniel C.S. Wilson, David Beavon, "Addressing bias and representativeness in digital heritage collections", _Digital Scholarship in the Humanities_, (forthcoming)

# Overview

- [Introduction](#introduction)
- [Data](#data)
    - [Press Directories](#press-directories)
        - [Processing](#processing)
        - [Description](#description)  
- [Code](#code)

## Introduction

## Data

### Press Directories
#### Processing
#### Description

### JISC

## Code

This repository contains a structured conversion of Mitchell's Newspaper Press Directories between 1846 and 1920. The code that transforms original scans to a tabular format is provided here as well and is documented in a series of Jupyter Notebooks that explain the different steps involved. 

A more elaborate description of the digitization process will be published in an upcoming publication. 

The press directories appeared almost yearly after 1846 and listed newspapers circulating in the United Kingdom, often giving detailled description of their audiences. 

The image below, shown a fragment for the `Greenwich Free Press` taking from the 1857 Press Directory, which we use as an example of the data description.

![Example of the Greenwich Free Press](img/example.png)

## Description of the dataset

**id**: unique identifier of the record, in the shape of MPD_{YEAR}_{POSITION}, i.e. `MPD_1846_3` is the third newspaper encountered in the 1846 edition

**TITLE**: The title of the newspaper, i.e. `Greenwich Free Press`

**POLITICS**: The political leaning of the newspaper, in this case `Liberal`, other frequent values are `Conservative`, `Neutral` or `Independent`

**CATEGORY**: The general category in which the newspaper appeared, in this case `provincial` as Greenwich wasn't absorbed by London yet in 1857. Other categories are `london`, `welsh`, `scottish`, `irish` and `isles` 

**DISTRICT_PUB**: District of publication `Greenwich`

**COUNTY_PUB**: County of publication `Kent`. The press directories are alphabetically ordered by district

**PRICE**: The prices indicated for this newspaper, if more than one price is mentioned, then these are joined by a `<SEP>` token, for example `1½d<SEP>2½d`. **Warning**: Prices are not processed further and need to converted in order to be comparable.

**ESTABLISHED_DATE**: Date the newspaper was establised, varies in precision, from an exact day to an approximate year. In this case the price date is give Aug 4. 1855. **Warning**: Dates are not normalized or converted to time-stamps, this would require further processing

**PUBLICATION_DAY**: The day of publication, often refers to a day in the week (e.g. `Thursday`), or specified in periodic terms such (`daily` or `weekly`). The `Greenwich Free Press` is published on `Saturday`.

**CIRCULATION**: Places where the newspaper was presumed to circulate, often gives a wider range than of places compared to `DISTICT_PUB`. Places are joined by a `<SEP>` token. In the running example this is: `Greenwich<SEP>Kent`

**TEXT_ORIGINAL**: The full text description for one newspaper. The other items mentioned here (such as price and politics) are extracted from this column. 

**DISTRICT_DESCRIPTION**: A textual description of the district (as captured by `DISTRICT_PUB` and `COUNTY_PUB` ). 

**YEAR**: Year of publication for the edition.

**NEWSPAPER_ID**: This is an unique identifier used to link newspaper over time. 

**WIKIDATA_ID**: The wikidata id for the place of publication.
**WIKIDATA_LINKING_RULE**: The rule used for linking place of publication to wikidata.
**LATITUDE**: latitude for the place of publication.
**LONGITUDE**: longitude for the place of publication.

## Digitization Process

Mitchell’s Press Directories were scanned by the British Library imaging studio in 2019 and further processed and enriched by the Living with Machines project.

The main goal of the enrichment was to transform scans to a structured format, which implied automatically recognizing the separate entries as well as parsing their content (i.e. finding the title, price and political leaning etc.). However, as often with heritage collections, this proved more difficult than initially anticipated. As the above figure shows, the typesetting of the directories follows a rather rigid pattern, but the varied use of fonts and types proved challenging for standard OCR software such as ABBYY.  Using Transfer Learning, a technique in which large models are fine-tuned on an often smaller, domain-specific (historical) dataset, we managed to improve the quality. We created our own pipeline that attunes each step to the particularities of the historical resource. The first step included image processing. We finetuned a UNet to convert images into columns and removed some features that caused trouble for the OCR software, for example the drop-down capitals. Then we improved a general Tesseract 5 model, by fine-tuning it on line-transcriptions from the press directories. 

The OCR was then further processed in two stages: page splitting and semantic annotation. The first part divides pages into their main components, i.e. the districts in which the newspapers circulated (with a sort of stub text that aims to characterize a place) and the newspaper entries. The second part then parses these profiles, to extract specific information, on top of which we added some further normalisation (unifying the political labels and prices and places of publication) and linking (associating places of publication outward with wikidata identifiers). Both parsing and semantic tagging was done by fine-tuning BERT models on a large set of manual annotations. Lastly, we created a record linker (also by training BERT on pairs of titles) which allowed us to follow a particular newspaper over time. 

Lastly, to improve the quality of the data, we performed a light-touch manual correction of the data for a selected number years, this to spot-check for serious errors (which fortunately didn’t appear) and correct titles and places of publication that proved difficult for the OCR engine to process. While not flawless we are confident that the data generated from the scans is reliable enough for the purposes of the environmental scan.
