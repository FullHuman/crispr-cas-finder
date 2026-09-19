# Third-party notices and data provenance

This independent Rust implementation is distributed under GPL-3.0-or-later.
The project license is in LICENSE. Upstream code/data retain their applicable
licenses and notices below. Upstream authors do not endorse this implementation.

## CRISPRCasFinder and CasFinder data

Source: https://github.com/dcouvin/CRISPRCasFinder
Reference revision: 9a7f07edc4001875be05f6f0f0eb81f4fdf52e4d
License: GPL-3.0-or-later; upstream COPYING contains the GPL v3 text in LICENSE.

The bundled CasFinder-2.0.3 XML definitions and HMM profiles were compared with
this revision: 152 of 153 files are byte-identical. CAS-TypeIIID.xml removes a
stray `/>` following the Csm2_1_IIIA gene element and one trailing blank line.
The Repeat_List.csv and repeatDirection.tsv files are byte-identical to the
same revision's supplementary_files. The browser model bundle is generated
from these local definitions/profiles. Preserve these notes when changing data.

Upstream copyright notice follows:

```text
CRISPRCasFinder allows to find CRISPR arrays as well as associated cas genes.

Copyright (C) 2017- CRISPR-Cas++ team (CNRS, Université Paris-Saclay, Institut Pasteur, Université de Lille).
------------------------------------------------------------------

Citation: 
Couvin D, Bernheim A, Toffano-Nioche C, Touchon M, Michalik J, Néron B, Rocha EPC, Vergnaud G, Gautheret D, Pourcel C. 
CRISPRCasFinder, an update of CRISRFinder, includes a portable version, enhanced performance and integrates search for Cas proteins. 
Nucleic Acids Res. 2018 Jul 2;46(W1):W246-W251. DOI: https://doi.org/10.1093/nar/gky425
------------------------------------------------------------------

All the contributors of the program are: 
David Couvin,
Aude Bernheim,
Claire Toffano-Nioche,
Marie Touchon,
Bertrand Néron,
Christine Drevet,
Luis M. Rodriguez,
Lee S. Katz,
Juraj Michalik,
Fabrice Leclerc,
Cyrille Petat,
Arnaud Martel,
Nicolas Villeriot.

-----------------------------------------------------------------
The source code of CRISPRCasFinder is distributed 
under the terms of the GNU GeneralPublic License. 
See the file COPYING for details.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
```

## HMMER and Easel portions

The internal modules src/hmmer_core and src/hmmer_io include Rust ports of
HMMER and Easel functionality, adapted and modified in this project. They are
not an official HMMER release. The historical port did not record a precise
upstream revision; the following revisions identify the preserved upstream
notice texts and reference implementation, not a claim of an exact source fork.

HMMER reference: 3.4, revision 9acd8b6758a0ca5d21db6d167e0277484341929b
https://github.com/EddyRivasLab/hmmer/blob/9acd8b6758a0ca5d21db6d167e0277484341929b/LICENSE

```text
HMMER - Biological sequence analysis with profile hidden Markov models
Copyright (C) 1992-2023 Sean R. Eddy
Copyright (C) 2015-2023 President and Fellows of Harvard College
Copyright (C) 2000-2023 Howard Hughes Medical Institute
Copyright (C) 1995-2006 Washington University School of Medicine
Copyright (C) 1992-1995 MRC Laboratory of Molecular Biology
-----------------------------------------------------------------------

The code includes contributions and input from current and past
members of the HMMER development team, as well as other colleagues and
sources, including:

   Bill Arndt        
   Jeremy Buhler     
   Tyler Camp
   Nick Carter        
   Sergi Castellano  
   Goran Ceric       
   Michael Farrar    
   Rob Finn          
   Ian Holmes        
   Bjarne Knudsen    
   Diana Kolbe       
   Martin Larralde
   Erik Lindahl      
   Graeme Mitchison  
   Eric Nawrocki     
   Lee Newberg       
   Sam Petti
   Elena Rivas        
   Walt Shands  
   Travis Wheeler    

HMMER also includes copyrighted and licensed code that has been
incorporated from other sources, including:

   Yuta Mori (libdivsufsort-lite)
   Apple Computer                  
   Free Software Foundation, Inc.  
   IBM TJ Watson Research Center   
   X Consortium                    

HMMER uses the Easel software library, which has its own license and
copyright information. See easel/LICENSE.

HMMER includes patent-pending SIMD technology under a nonexclusive
license from the estate of Michael Farrar. You are sublicensed to use
this technology specifically for the use, modification, and
redistribution of HMMER.

HMMER development is supported in part by the National Human Genome
Research Institute of the US National Institutes of Health under grant
number R01HG009116. The content is solely the responsibility of the
authors and does not necessarily represent the official views of the
National Institutes of Health.

HMMER source code is distributed as open source under the terms of the
BSD three-clause license:

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

   3. Neither the name of any copyright holder nor the names of 
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
   FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
   COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
   STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
   OF THE POSSIBILITY OF SUCH DAMAGE.






   

```

Easel notice revision: 07ca83ba9ef0414dba9ce0a9331d465b5eb58f2b
https://github.com/EddyRivasLab/easel/blob/07ca83ba9ef0414dba9ce0a9331d465b5eb58f2b/LICENSE

```text
Easel - a library of C functions for biological sequence analysis
Copyright (C) 1990-2023 Sean R. Eddy
Copyright (C) 2015-2023 President and Fellows of Harvard College
Copyright (C) 2000-2023 Howard Hughes Medical Institute
Copyright (C) 1995-2006 Washington University School of Medicine
Copyright (C) 1992-1995 MRC Laboratory of Molecular Biology, UK
-------------------------------------------------------------------

Easel's code includes contributions from members of the Eddy/Rivas
laboratory at Harvard University, the HMMER and Infernal development
teams, and other colleagues, including:

   Tyler Camp        
   Nick Carter       
   Michael Farrar    
   Graeme Mitchison  
   Eric Nawrocki     
   Sam Petti
   Elena Rivas       
   Travis Wheeler    

Easel also includes code we have incorporated from other sources and
authors -- including public domain code, and licensed copyrighted
code. Sources and licenses are noted in the appropriate places in
individual files. Copyright holders and contributors include:

   Barry W. Brown, James Lovato        esl_random:esl_rnd_Gaussian()   
   Bob Jenkins                         esl_random::esl_rnd_mix3()
   Steven G. Johnson, and others       autoconf macros in m4/
   Martin Larralde                     ARM Neon support
   Kevin Lawler                        esl_rand64::esl_rand64_Deal()
   Stephen Moshier                     SIMD vectorized logf,expf
   Takuji Nishimura, Makoto Matsumoto  esl_random, esl_rand64
   Julien Pommier                      SIMD vectorized logf,expf
   David Robert Nadeau                 esl_stopwatch
   Henry Spencer                       esl_regexp
   David Wheeler                       easel::esl_tmpfile()
   Free Software Foundation, Inc.      configure	
   FreeBSD 		               easel::esl_strsep()
   Sun Microsystems, Inc.              esl_stats::esl_erfc()

Easel development is supported in part by the National Human Genome
Research Institute of the US National Institutes of Health under grant
number R01HG009116. The content is solely the responsibility of the
authors and does not necessarily represent the official views of the
National Institutes of Health.

The Easel library is open source software. It is freely distributed
under the terms of the BSD (Berkeley Software Distribution) open
source license:

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
   FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
   COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
   STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
   OF THE POSSIBILITY OF SUCH DAMAGE.


```

## Orphos and other dependencies

Gene prediction uses orphos-core 0.3.0, distributed under GPL-3.0-or-later:
https://github.com/FullHuman/orphos. Other Rust dependency versions and sources
are recorded in the accompanying Cargo.lock files and retain their own licenses.
These notices document the bundled scientific code and datasets; they do not
relicense third-party dependencies or assert authorship of upstream material.
