* ALE(Arcade-Learning-Environment) 3rd party library(?) 설명
 (1) 본 라이브러리(?)의 라이센스는 GPL v2이다.
 (2) include, src, rom 3개의 폴더로 구성 된다.
   - include폴더는 src, src/controller, src/os_dependent, src/environment, src/external의 
    헤더파일들(.h, .hxx, .hpp)을 복사 하였다.
   - 이것은 doc/examples/Makefile.sharedlibrary 파일을 기준으로 하였으며 부족하다면 추후에
    추가 한다.
 (3) shared library는 Linux 64bit 환경에서 빌드하였다.
 (4) rom폴더에는 Atari 2600의 롬파일들을 복사 하였다. 
   - 아래 사이트에서 무료로 다운 받았다. 라이센스 정책에 대한 확인이 필요하다.
     https://www.atariage.com/system_items.html?SystemID=2600&ItemTypeID=ROM
   - Atari 2600의 모든 롬파일이 동작하는 것은 아니다. 아래의 사이트에서는 ALE가 지원하는 롬
   파일들을 열거해 놓았다. 하지만, 열거된 롬파일들의 일부는 동작하지 않았다.
     http://yavar.naddaf.name/ale/list_of_current_games.html
   - 정상 동작하는 롬파일에 대해서 복사 하였다. 
    (현재 3개, ALEIN.BIN, Bowling.bin, Breakout.bin)
 (5) http://www.arcadelearningenvironment.org/에서 소스를 받았고, 0.5.1버전이다.
 (6) external/TinyMT, emucore/rsynth, emucore/m6502, emucore/m6502/src/bspf에도 따로 라이센스가 있다.
