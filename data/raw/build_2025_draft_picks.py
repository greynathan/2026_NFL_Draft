"""One-off: build data/raw/2025_draft_picks.csv from PFR 2025 draft table (Rnd,Pick,Tm,Player,Pos,School)."""
import csv
import os

# PFR table rows: Rnd,Pick,Tm,Player,Pos,Age,To,...,College/Univ,College Stats,id
# We take indices 0,1,2,3,4 and -3 (School)
RAW_TABLE = """1,1,TEN,Cam Ward,QB,23,2025,0,0,0,7,7,17,323,540,3169,15,7,39,159,2,0,0,0,,,,Miami (FL),College Stats,WardCa00
1,2,JAX,Travis Hunter,WR,22,2025,0,0,0,5,5,7,0,0,0,0,0,1,0,0,28,298,1,11,,,Colorado,College Stats,HuntTr00
1,3,NYG,Abdul Carter,DE,21,2025,0,0,0,4,4,17,0,0,0,0,0,0,0,0,0,0,0,25,,4.0,Penn St.,College Stats,CartAb00
1,4,NWE,Will Campbell,OT,21,2025,0,0,0,8,8,13,0,0,0,0,0,0,0,0,0,0,0,,,,LSU,College Stats,CampWi01
1,5,CLE,Mason Graham,DT,22,2025,0,0,0,8,8,17,0,0,0,0,0,0,0,0,0,0,0,28,,0.5,Michigan,College Stats,GrahMa00
1,6,LVR,Ashton Jeanty,RB,21,2025,0,0,0,7,7,17,0,0,0,0,0,266,975,5,55,346,5,,,,Boise St.,College Stats,JeanAs00
1,7,NYJ,Armand Membou,OL,21,2025,0,0,0,7,7,17,0,0,0,0,0,0,0,0,0,0,0,,,,Missouri,College Stats,MembAr00
1,8,CAR,Tetairoa McMillan,WR,22,2025,0,0,0,9,9,17,0,0,0,0,0,0,0,0,70,1014,7,,,,Arizona,College Stats,McMiTe00
1,9,NOR,Kelvin Banks,OL,21,2025,0,0,0,6,6,17,0,0,0,0,0,0,0,0,0,0,0,,,,Texas,College Stats,BankKe01
1,10,CHI,Colston Loveland,TE,21,2025,0,0,0,7,7,16,0,0,0,0,0,1,-2,0,58,713,6,,,,Michigan,College Stats,LoveCo00
1,11,SFO,Mykel Williams,DL,21,2025,0,0,0,4,4,9,0,0,0,0,0,0,0,0,0,0,0,11,,1.0,Georgia,College Stats,WillMy00
1,12,DAL,Tyler Booker,OL,21,2025,0,0,0,6,6,14,0,0,0,0,0,0,0,0,0,0,0,,,,Alabama,College Stats,BookTy00
1,13,MIA,Kenneth Grant,DT,21,2025,0,0,0,3,3,17,0,0,0,0,0,0,0,0,0,0,0,15,,2.0,Michigan,College Stats,GranKe01
1,14,IND,Tyler Warren,TE,23,2025,0,0,0,9,9,17,0,1,0,0,0,6,8,1,76,817,4,,,,Penn St.,College Stats,WarrTy00
1,15,ATL,Jalon Walker,DE,21,2025,0,0,0,5,5,15,0,0,0,0,0,0,0,0,0,0,0,24,,5.5,Georgia,College Stats,WalkJa02
1,16,ARI,Walter Nolen,DT,21,2025,0,0,0,1,1,6,0,0,0,0,0,0,0,0,0,0,0,8,,2.0,Mississippi,College Stats,NoleWa00
1,17,CIN,Shemar Stewart,DE,21,2025,0,0,0,2,2,8,0,0,0,0,0,0,0,0,0,0,0,5,,1.0,Texas A&M,College Stats,StewSh00
1,18,SEA,Grey Zabel,OT,23,2025,0,0,0,7,7,17,0,0,0,0,0,0,0,0,0,0,0,,,,North Dakota St.,,ZabeGr00
1,19,TAM,Emeka Egbuka,WR,22,2025,0,0,0,9,9,17,0,0,0,0,0,2,9,0,63,938,6,,,,Ohio St.,College Stats,EgbuEm01
1,20,DEN,Jahdae Barron,CB,23,2025,0,0,0,3,3,17,0,0,0,0,0,0,0,0,0,0,0,24,1,,Texas,College Stats,BarrJa01
1,21,PIT,Derrick Harmon,DT,22,2025,0,0,0,4,4,12,0,0,0,0,0,0,0,0,0,0,0,11,,3.0,Oregon,College Stats,HarmDe01
1,22,LAC,Omarion Hampton,RB,22,2025,0,0,0,5,5,9,0,0,0,0,0,124,545,4,32,192,1,,,,North Carolina,College Stats,HampOm00
1,23,GNB,Matthew Golden,WR,22,2025,0,0,0,4,4,14,0,0,0,0,0,10,49,0,29,361,0,,,,Texas,College Stats,GoldMa02
1,24,MIN,Donovan Jackson,OL,22,2025,0,0,0,5,5,14,0,0,0,0,0,0,0,0,0,0,0,,,,Ohio St.,College Stats,JackDo03
1,25,NYG,Jaxson Dart,QB,22,2025,0,0,0,11,11,14,216,339,2272,15,5,86,487,9,0,0,0,,,,Mississippi,College Stats,DartJa00
1,26,ATL,James Pearce,DE,21,2025,0,0,0,3,3,17,0,0,0,0,0,0,0,0,0,0,0,17,,10.5,Tennessee,College Stats,PearJa00
1,27,BAL,Malaki Starks,SAF,21,2025,0,0,0,5,5,17,0,0,0,0,0,0,0,0,0,0,0,49,2,,Georgia,College Stats,StarMa00
1,28,DET,Tyleik Williams,DT,22,2025,0,0,0,4,4,17,0,0,0,0,0,0,0,0,0,0,0,8,,1.0,Ohio St.,College Stats,WillTy02
1,29,WAS,Josh Conerly,OL,21,2025,0,0,0,8,8,17,0,0,0,0,0,0,0,0,0,0,0,,,,Oregon,College Stats,ConeJo00
1,30,BUF,Maxwell Hairston,CB,22,2025,0,0,0,2,2,11,0,0,0,0,0,0,0,0,0,0,0,14,2,,Kentucky,College Stats,HairMa00
1,31,PHI,Jihaad Campbell,LB,21,2025,0,0,0,6,6,17,0,0,0,0,0,0,0,0,0,0,0,44,1,,Alabama,College Stats,CampJi00
1,32,KAN,Josh Simmons,OL,22,2025,0,0,0,4,4,8,0,0,0,0,0,0,0,0,0,0,0,,,,Ohio St.,College Stats,SimmJo01
2,33,CLE,Carson Schwesinger,LB,22,2025,0,0,0,8,8,16,0,0,0,0,0,0,0,0,0,0,0,67,2,2.5,UCLA,College Stats,SchwCa00
2,34,HOU,Jayden Higgins,WR,22,2025,0,0,0,4,4,17,0,0,0,0,0,0,0,0,41,525,6,,,,Iowa St.,College Stats,HiggJa00
2,35,SEA,Nick Emmanwori,SAF,21,2025,0,0,0,4,4,14,0,0,0,0,0,0,0,0,0,0,0,56,1,2.5,South Carolina,College Stats,EmmaNi00
2,36,CLE,Quinshon Judkins,RB,21,2025,0,0,0,4,4,14,0,0,0,0,0,230,827,7,26,171,0,,,,Ohio St.,College Stats,JudkQu00
2,37,MIA,Jonah Savaiinaea,OL,21,2025,0,0,0,6,6,17,0,0,0,0,0,0,0,0,0,0,0,,,,Arizona,College Stats,SavaJo00
2,38,NWE,TreVeyon Henderson,RB,22,2025,0,0,0,8,8,17,0,0,0,0,0,180,911,9,35,221,1,,,,Ohio St.,College Stats,HendTr02
2,39,CHI,Luther Burden,WR,21,2025,0,0,0,6,6,15,0,0,0,0,0,6,37,0,47,652,2,,,,Missouri,College Stats,BurdLu00
2,40,NOR,Tyler Shough,QB,25,2025,0,0,0,7,7,11,221,327,2384,10,6,45,186,3,0,0,0,,,,Louisville,College Stats,ShouTy00
2,41,BUF,T.J. Sanders,DT,22,2025,0,0,0,2,2,12,0,0,0,0,0,0,0,0,0,0,0,7,,1.0,South Carolina,College Stats,SandTJ00
2,42,NYJ,Mason Taylor,TE,21,2025,0,0,0,3,3,13,0,0,0,0,0,0,0,0,44,369,1,,,,LSU,College Stats,TaylMa02
2,43,SFO,Alfred Collins,DT,23,2025,0,0,0,2,2,16,0,0,0,0,0,0,0,0,0,0,0,4,,1.0,Texas,College Stats,CollAl01
2,44,DAL,Donovan Ezeiruaku,DE,21,2025,0,0,0,3,3,17,0,0,0,0,0,0,0,0,0,0,0,20,,2.0,Boston Col.,College Stats,EzeiDo00
2,45,IND,JT Tuimoloau,DE,22,2025,0,0,0,1,1,13,0,0,0,0,0,0,0,0,0,0,0,6,,,Ohio St.,College Stats,TuimJT00
2,46,LAR,Terrance Ferguson,TE,22,2025,0,0,0,2,2,14,0,0,0,0,0,1,0,0,11,231,3,,,,Oregon,College Stats,FergTe00
2,47,ARI,Will Johnson,CB,22,2025,0,0,0,3,3,12,0,0,0,0,0,0,0,0,0,0,0,27,,,Michigan,College Stats,JohnWi02
2,48,HOU,Aireontae Ersery,OL,23,2025,0,0,0,7,7,16,0,0,0,0,0,0,0,0,0,0,0,,,,Minnesota,College Stats,ErseAi00
2,49,CIN,Demetrius Knight,LB,25,2025,0,0,0,6,6,17,0,0,0,0,0,0,0,0,0,0,0,58,2,3.0,South Carolina,College Stats,KnigDe00
2,50,SEA,Elijah Arroyo,TE,22,2025,0,0,0,2,2,13,0,0,0,0,0,0,0,0,15,179,1,,,,Miami (FL),College Stats,ArroEl00
2,51,CAR,Nic Scourton,DE,21,2025,0,0,0,4,4,17,0,0,0,0,0,0,0,0,0,0,0,20,,5.0,Texas A&M,College Stats,ScouNi00
2,52,TEN,Oluwafemi Oladejo,DE,21,2025,0,0,0,1,1,6,0,0,0,0,0,0,0,0,0,0,0,5,,,UCLA,College Stats,OladOl00
2,53,TAM,Benjamin Morrison,CB,21,2025,0,0,0,2,2,10,0,0,0,0,0,0,0,0,0,0,0,19,,,Notre Dame,College Stats,MorrBe00
2,54,GNB,Anthony Belton,OT,24,2025,0,0,0,5,5,14,0,0,0,0,0,0,0,0,0,0,0,,,,North Carolina St.,College Stats,BeltAn00
2,55,LAC,Tre Harris,WR,23,2025,0,0,0,3,3,17,0,0,0,0,0,2,10,0,30,324,1,,,,Mississippi,College Stats,HarrTr03
2,56,CHI,Ozzy Trapilo,OL,23,2025,0,0,0,3,3,14,0,0,0,0,0,0,0,0,0,0,0,,,,Boston Col.,College Stats,TrapOz00
2,57,DET,Tate Ratledge,OL,24,2025,0,0,0,7,7,17,0,0,0,0,0,0,0,0,0,0,0,,,,Georgia,College Stats,RatlTa00
2,58,LVR,Jack Bech,WR,22,2025,0,0,0,2,2,16,0,0,0,0,0,0,0,0,20,224,0,1,,,TCU,College Stats,BechJa00
2,59,BAL,Mike Green,DE,22,2025,0,0,0,2,2,17,0,0,0,0,0,0,0,0,0,0,0,17,,3.5,Marshall,College Stats,GreeMi02
2,60,DEN,RJ Harvey,RB,24,2025,0,0,0,6,6,17,0,1,0,0,0,146,540,7,47,356,5,,,,Central Florida,College Stats,HarvRJ00
2,61,WAS,Trey Amos,CB,23,2025,0,0,0,2,2,10,0,0,0,0,0,0,0,0,0,0,0,19,,,Mississippi,College Stats,AmosTr00
2,62,CHI,Shemar Turner,DE,22,2025,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,6,,,Texas A&M,College Stats,TurnSh00
2,63,KAN,Omarr Norman-Lott,DT,23,2025,0,0,0,1,1,5,0,0,0,0,0,0,0,0,0,0,0,4,,1.0,Tennessee,College Stats,NormOm00
2,64,PHI,Andrew Mukuba,SAF,22,2025,0,0,0,3,3,11,0,0,0,0,0,0,0,0,0,0,0,27,2,0.5,Texas,College Stats,MukuAn00
3,65,NYG,Darius Alexander,DT,25,2025,0,0,0,2,2,16,0,0,0,0,0,0,0,0,0,0,0,9,,3.5,Toledo,College Stats,AlexDa01
3,66,KAN,Ashton Gillotte,DE,22,2025,0,0,0,3,3,17,0,0,0,0,0,0,0,0,0,0,0,14,1,1.5,Louisville,College Stats,GillAs00
3,67,CLE,Harold Fannin,TE,21,2025,0,0,0,5,5,16,0,0,0,0,0,7,13,1,72,731,6,,,,Bowling Green,College Stats,FannHa00
3,68,LVR,Darien Porter,CB,24,2025,0,0,0,4,4,17,0,0,0,0,0,0,0,0,0,0,0,29,,,Iowa St.,College Stats,PortDa00
3,69,NWE,Kyle Williams,WR,22,2025,0,0,0,2,2,17,0,0,0,0,0,2,3,0,10,209,3,,,,Washington St.,College Stats,WillKy03
3,70,DET,Isaac TeSlaa,WR,23,2025,0,0,0,2,2,17,0,0,0,0,0,0,0,0,16,239,6,,,,Arkansas,College Stats,TeSlIs00
3,71,NOR,Vernon Broughton,DT,24,2025,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,,,,Texas,College Stats,BrouVe00
3,72,BUF,Landon Jackson,DE,22,2025,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,,,,Arkansas,College Stats,JackLa03
3,73,NYJ,Azareye'h Thomas,CB,21,2025,0,0,0,2,2,12,0,0,0,0,0,0,0,0,0,0,0,13,,,Florida St.,College Stats,ThomAz00
3,74,DEN,Pat Bryant,WR,22,2025,0,0,0,3,3,15,0,0,0,0,0,0,0,0,31,378,1,,,,Illinois,College Stats,BryaPa00
3,75,SFO,Nick Martin,LB,22,2025,0,0,0,1,1,7,0,0,0,0,0,0,0,0,0,0,0,8,,,Oklahoma St.,College Stats,MartNi01
3,76,DAL,Shavon Revel,CB,24,2025,0,0,0,1,1,7,0,0,0,0,0,0,0,0,0,0,0,23,,,East Carolina,College Stats,ReveSh00
3,77,CAR,Princely Umanmielen,DE,23,2025,0,0,0,2,2,16,0,0,0,0,0,0,0,0,0,0,0,13,,1.5,Mississippi,College Stats,UmanPr00
3,78,ARI,Jordan Burch,DE,23,2025,0,0,0,1,1,17,0,0,0,0,0,0,0,0,0,0,0,5,,1.0,Oregon,College Stats,BurcJo00
3,79,HOU,Jaylin Noel,WR,23,2025,0,0,0,2,2,17,0,0,0,0,0,6,12,0,26,292,2,,,,Iowa St.,College Stats,NoelJa00
3,80,IND,Justin Walley,CB,22,,0,0,0,,,,,,,,,,,,,,,,,,Minnesota,College Stats,WallJu00
3,81,CIN,Dylan Fairchild,OL,22,2025,0,0,0,6,6,15,0,0,0,0,0,0,0,0,0,0,0,,,,Georgia,College Stats,FairDy00
3,82,TEN,Kevin Winston,SAF,21,2025,0,0,0,2,2,10,0,0,0,0,0,0,0,0,0,0,0,21,,1.0,Penn St.,College Stats,WinsKe02
3,83,PIT,Kaleb Johnson,RB,22,2025,0,0,0,1,1,10,0,0,0,0,0,28,69,0,1,9,0,,,,Iowa,College Stats,JohnKa03
3,84,TAM,Jacob Parrish,CB,21,2025,0,0,0,3,3,17,0,0,0,0,0,0,0,0,0,0,0,50,2,2.0,Kansas St.,College Stats,ParrJa01
3,85,KAN,Nohl Williams,CB,22,2025,0,0,0,3,3,17,0,0,0,0,0,0,0,0,0,0,0,30,,1.0,California,College Stats,WillNo01
3,86,LAC,Jamaree Caldwell,DT,25,2025,0,0,0,3,3,17,0,0,0,0,0,0,0,0,0,0,0,14,,1.0,Oregon,College Stats,CaldJa00
3,87,GNB,Savion Williams,WR,23,2025,0,0,0,1,1,12,0,0,0,0,0,11,37,0,10,78,1,,,,TCU,College Stats,WillSa01
3,88,JAX,Caleb Ransaw,SAF,22,,0,0,0,,,,,,,,,,,,,,,,,,Tulane,College Stats,RansCa00
3,89,JAX,Wyatt Milum,OL,22,2025,0,0,0,1,1,10,0,0,0,0,0,0,0,0,0,0,0,,,,West Virginia,College Stats,MiluWy00
3,90,LAR,Josaiah Stewart,DE,22,2025,0,0,0,2,2,17,0,0,0,0,0,0,0,0,0,0,0,13,,3.0,Michigan,College Stats,StewJo03
3,91,BAL,Emery Jones,OL,21,2025,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,,,,LSU,College Stats,JoneEm02
3,92,SEA,Jalen Milroe,QB,22,2025,0,0,0,0,0,3,0,0,0,0,0,3,4,0,0,0,0,,,,Alabama,College Stats,MilrJa00
3,93,NOR,Jonas Sanker,SAF,22,2025,0,0,0,7,7,17,0,0,0,0,0,0,0,0,0,0,0,46,2,,Virginia,College Stats,SankJo00
3,94,CLE,Dillon Gabriel,QB,24,2025,0,0,0,0,0,10,110,185,937,7,2,14,86,0,0,0,0,,,,Oregon,College Stats,GabrDi00
3,95,NWE,Jared Wilson,OL,22,2025,0,0,0,7,7,13,0,0,0,0,0,0,0,0,0,0,0,,,,Georgia,College Stats,WilsJa02
3,96,ATL,Xavier Watts,SAF,23,2025,0,0,0,7,7,17,0,0,0,0,0,0,0,0,0,0,0,59,5,,Notre Dame,College Stats,WattXa00
3,97,HOU,Jaylin Smith,CB,21,2025,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,3,,,USC,College Stats,SmitJa11
3,98,LVR,Caleb Rogers,OL,23,2025,0,0,0,2,2,6,0,0,0,0,0,0,0,0,0,0,0,,,,Texas Tech,College Stats,RogeCa01
3,99,LVR,Charles Grant,OL,23,2025,0,0,0,1,1,9,0,0,0,0,0,0,0,0,0,0,0,,,,William & Mary,,GranCh00
3,100,SFO,Upton Stout,CB,23,2025,0,0,0,2,2,16,0,0,0,0,0,0,0,0,0,0,0,43,,1.0,Western Kentucky,College Stats,StouUp00
3,101,DEN,Sai'vion Jones,DE,22,2025,0,0,0,1,1,3,0,0,0,0,0,0,0,0,0,0,0,,,,LSU,College Stats,JoneSa01
3,102,MIN,Tai Felton,WR,22,2025,0,0,0,0,0,17,0,0,0,0,0,0,0,0,3,25,0,5,,,Maryland,College Stats,FeltTa00
4,103,TEN,Chimere Dike,WR,23,2025,1,1,0,5,5,17,0,0,0,0,0,11,18,0,48,423,4,,,,Florida,College Stats,DikeCh00
4,104,JAX,Bhayshul Tuten,RB,22,2025,0,0,0,3,3,15,0,0,0,0,0,83,307,5,10,79,2,,,,Virginia Tech,College Stats,TuteBh00
4,105,NYG,Cam Skattebo,RB,23,2025,0,0,0,5,5,8,0,0,0,0,0,101,410,5,24,207,2,,,,Arizona St.,College Stats,SkatCa00
4,106,NWE,Craig Woodson,SAF,24,2025,0,0,0,6,6,17,0,0,0,0,0,0,0,0,0,0,0,42,,,California,College Stats,WoodCr00
4,107,JAX,Jack Kiser,LB,24,2025,0,0,0,1,1,14,0,0,0,0,0,0,0,0,0,0,0,5,,,Notre Dame,College Stats,KiseJa00
4,108,LVR,Dont'e Thornton,WR,22,2025,0,0,0,1,1,15,0,0,0,0,0,0,0,0,10,135,0,,,,Tennessee,College Stats,ThorDo00
4,109,BUF,Deone Walker,DT,21,2025,0,0,0,8,8,17,0,0,0,0,0,0,0,0,0,0,0,17,,1.0,Kentucky,College Stats,WalkDe04
4,110,NYJ,Arian Smith,WR,23,2025,0,0,0,1,1,16,0,0,0,0,0,4,13,0,7,52,0,3,,,Georgia,College Stats,SmitAr01
4,111,PHI,Ty Robinson,DT,24,2025,0,0,0,1,1,11,0,0,0,0,0,0,0,0,0,0,0,1,,,Nebraska,College Stats,RobiTy02
4,112,NOR,Danny Stutsman,LB,22,2025,0,0,0,1,1,17,0,0,0,0,0,0,0,0,0,0,0,25,,,Oklahoma,College Stats,StutDa00
4,113,SFO,CJ West,DT,23,2025,0,0,0,1,1,14,0,0,0,0,0,0,0,0,0,0,0,12,,1.0,Indiana,College Stats,WestCJ00
4,114,CAR,Trevor Etienne,RB,21,2025,0,0,0,1,1,17,0,0,0,0,0,20,94,0,3,13,0,,,,Georgia,College Stats,EtieTr01
4,115,ARI,Cody Simon,LB,23,2025,0,0,0,3,3,16,0,0,0,0,0,0,0,0,0,0,0,30,,0.5,Ohio St.,College Stats,SimoCo01
4,116,HOU,Woody Marks,RB,24,2025,0,0,0,6,6,16,0,0,0,0,0,196,703,2,24,208,3,,,,USC,College Stats,MarkWo00
4,117,LAR,Jarquez Hunter,RB,22,2025,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,,,,Auburn,College Stats,HuntJa03
4,118,ATL,Billy Bowman,SAF,22,2025,0,0,0,1,1,6,0,0,0,0,0,0,0,0,0,0,0,15,1,1.5,Oklahoma,College Stats,BowmBi01
4,119,CIN,Barrett Carter,LB,22,2025,0,0,0,5,5,17,0,0,0,0,0,0,0,0,0,0,0,53,1,,Clemson,College Stats,CartBa00
4,120,TEN,Gunnar Helm,TE,23,2025,0,0,0,3,3,16,0,0,0,0,0,0,0,0,44,357,2,1,,,Texas,College Stats,HelmGu00
4,121,TAM,David Walker,OLB,25,,0,0,0,,,,,,,,,,,,,,,,,,Central Arkansas,,WalkDa03
4,122,CAR,Lathan Ransom,SAF,23,2025,0,0,0,3,3,16,0,0,0,0,0,0,0,0,0,0,0,21,1,1.0,Ohio St.,College Stats,RansLa00
4,123,PIT,Jack Sawyer,DE,23,2025,0,0,0,2,2,17,0,0,0,0,0,0,0,0,0,0,0,16,2,1.0,Ohio St.,College Stats,SawyJa00
4,124,GNB,Barryn Sorrell,DE,22,2025,0,0,0,2,2,14,0,0,0,0,0,0,0,0,0,0,0,7,,1.5,Texas,College Stats,SorrBa00
4,125,LAC,Kyle Kennard,DE,23,2025,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,1,,,South Carolina,College Stats,KennKy00
4,126,CLE,Dylan Sampson,RB,20,2025,0,0,0,3,3,15,0,0,0,0,0,65,175,0,33,271,2,,,,Tennessee,College Stats,SampDy00
4,127,IND,Jalen Travis,OT,23,2025,0,0,0,3,3,17,0,0,0,0,0,0,0,0,0,0,0,,,,Iowa St.,College Stats,TravJa00
4,128,WAS,Jaylin Lane,WR,23,2025,0,0,0,4,4,15,0,0,0,0,0,1,7,0,16,225,0,,,,Virginia Tech,College Stats,LaneJa00
4,129,BAL,Teddye Buchanan,LB,22,2025,0,0,0,5,5,14,0,0,0,0,0,0,0,0,0,0,0,49,,0.5,California,College Stats,BuchTe00
4,130,NYJ,Malachi Moore,SAF,23,2025,0,0,0,4,4,17,0,0,0,0,0,0,0,0,1,3,0,56,,,Alabama,College Stats,MoorMa04
4,131,NOR,Quincy Riley,CB,24,2025,0,0,0,3,3,17,0,0,0,0,0,0,0,0,0,0,0,29,1,,Louisville,College Stats,RileQu00
4,132,CHI,Ruben Hyppolite,LB,23,2025,0,0,0,1,1,7,0,0,0,0,0,0,0,0,0,0,0,3,,,Maryland,College Stats,HyppRu00
4,133,KAN,Jalen Royals,WR,22,2025,0,0,0,0,0,7,0,0,0,0,0,0,0,0,2,4,0,1,,,Utah St.,College Stats,RoyaJa00
4,134,DEN,Quandarrius Robinson,LB,24,2025,0,0,0,1,1,6,0,0,0,0,0,0,0,0,0,0,0,6,,0.5,Alabama,College Stats,RobiQu00
4,135,LVR,Tonka Hemingway,DT,23,2025,0,0,0,1,1,9,0,0,0,0,0,0,0,0,0,0,0,8,,4.0,South Carolina,College Stats,HemiTo00
4,136,TEN,Elic Ayomanor,WR,22,2025,0,0,0,4,4,16,0,0,0,0,0,0,0,0,41,515,4,,,,Stanford,College Stats,AyomEl00
4,137,NWE,Joshua Farmer,DT,22,2025,0,0,0,3,3,13,0,0,0,0,0,0,0,0,0,0,0,7,,,Florida St.,College Stats,FarmJo00
4,138,SFO,Jordan Watkins,WR,23,2025,0,0,0,0,0,4,0,0,0,0,0,0,0,0,2,26,0,,,,Mississippi,College Stats,WatkJo00
5,139,MIN,Tyrion Ingram-Dawkins,DE,22,2025,0,0,0,2,2,14,0,0,0,0,0,0,0,0,0,0,0,6,,1.0,Georgia,College Stats,IngrTy00
5,140,CAR,Cam Jackson,DT,23,2025,0,0,0,1,1,9,0,0,0,0,0,0,0,0,0,0,0,2,,,Florida,College Stats,JackCa01
5,141,BAL,Carson Vinson,OT,23,2025,0,0,0,1,1,7,0,0,0,0,0,0,0,0,0,0,0,,,,Alabama A&M,,VinsCa00
5,142,SEA,Rylie Mills,DL,24,2025,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,1,,,Notre Dame,College Stats,MillRy02
5,143,MIA,Jordan Phillips,DL,21,2025,0,0,0,6,6,17,0,0,0,0,0,0,0,0,0,0,0,8,,,Maryland,College Stats,PhilJo02
5,144,CLE,Shedeur Sanders,QB,23,2025,0,1,0,2,2,8,120,212,1400,7,10,21,169,1,0,0,0,,,,Colorado,College Stats,SandSh00
5,145,PHI,Mac McWilliams,CB,23,2025,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,1,,,Central Florida,College Stats,McWiMa01
5,146,NWE,Bradyn Swinson,DE,23,2025,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,1,,,LSU,College Stats,SwinBr01
5,147,SFO,Jordan James,RB,21,2025,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,,,,Oregon,College Stats,JameJo01
5,148,LAR,Ty Hamilton,DT,23,2025,0,0,0,1,1,14,0,0,0,0,0,0,0,0,0,0,0,1,,,Ohio St.,College Stats,HamiTy00
5,149,DAL,Jaydon Blue,RB,21,2025,0,0,0,1,1,5,0,0,0,0,0,38,129,1,1,5,0,,,,Texas,College Stats,BlueJa01
5,150,MIA,Jason Marshall,CB,23,2025,0,0,0,1,1,12,0,0,0,0,0,0,0,0,0,0,0,14,1,,Florida,College Stats,MarsJa00
5,151,IND,DJ Giddens,RB,22,2025,0,0,0,1,1,9,0,0,0,0,0,26,96,0,0,0,0,,,,Kansas St.,College Stats,GiddDJ00
5,152,DAL,Shemar James,LB,21,2025,0,0,0,2,2,14,0,0,0,0,0,0,0,0,0,0,0,40,,1.5,Florida,College Stats,JameSh00
5,153,CIN,Jalen Rivers,OL,22,2025,0,0,0,3,3,16,0,0,0,0,0,0,0,0,0,0,0,,,,Miami (FL),College Stats,RiveJa00
5,154,NYG,Marcus Mbow,OL,22,2025,0,0,0,2,2,13,0,0,0,0,0,0,0,0,0,0,0,,,,Purdue,College Stats,MbowMa00
5,155,MIA,Dante Trader,SAF,22,2025,0,0,0,2,2,17,0,0,0,0,0,0,0,0,0,0,0,29,,,Maryland,College Stats,TradDa00
5,156,KAN,Jeffrey Bassa,LB,22,2025,0,0,0,1,1,17,0,0,0,0,0,0,0,0,0,0,0,6,,,Oregon,College Stats,BassJe00
5,157,TAM,Elijah Roberts,DT,23,2025,0,0,0,5,5,17,0,0,0,0,0,0,0,0,0,0,0,10,,2.0,SMU,College Stats,RobeEl01
5,158,LAC,KeAndre Lambert-Smith,WR,23,2025,0,0,0,0,0,10,0,0,0,0,0,0,0,0,5,51,1,,,,Auburn,College Stats,LambKe01
5,159,GNB,Collin Oliver,DE,22,2025,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,,,Oklahoma St.,College Stats,OlivCo00
5,160,SFO,Marques Sigle,SAF,23,2025,0,0,0,3,3,15,0,0,0,0,0,0,0,0,0,0,0,28,,,Kansas St.,College Stats,SiglMa00
5,161,PHI,Smael Mondon,LB,22,2025,0,0,0,1,1,17,0,0,0,0,0,0,0,0,0,0,0,4,,,Georgia,College Stats,MondSm01
5,162,NYJ,Francisco Mauigoa,LB,22,2025,0,0,0,3,3,12,0,0,0,0,0,0,0,0,0,0,0,16,,,Miami (FL),College Stats,MauiFr00
5,163,CAR,Mitchell Evans,TE,22,2025,0,0,0,2,2,17,0,0,0,0,0,1,1,0,19,171,2,,,,Notre Dame,College Stats,EvanMi01
5,164,PIT,Yahya Black,DT,23,2025,0,0,0,2,2,17,0,0,0,0,0,0,0,0,0,0,0,10,,,Iowa,College Stats,BlacYa00
5,165,LAC,Oronde Gadsden II,TE,22,2025,0,0,0,6,6,15,0,0,0,0,0,0,0,0,49,664,3,,,,Syracuse,College Stats,GadsOr01
5,166,SEA,Tory Horton,WR,22,2025,0,0,0,2,2,8,0,0,0,0,0,0,0,0,13,161,5,,,,Colorado St.,College Stats,HortTo00
5,167,TEN,Jackson Slater,OL,22,2025,0,0,0,1,1,12,0,0,0,0,0,0,0,0,0,0,0,,,,Sacramento St.,,SlatJa01
5,168,PHI,Drew Kendall,C,23,2025,0,0,0,1,1,7,0,0,0,0,0,0,0,0,0,0,0,,,,Boston Col.,College Stats,KendDr00
5,169,CHI,Zah Frazier,CB,24,,0,0,0,,,,,,,,,,,,,,,,,,Texas-San Antonio,College Stats,FrazZa01
5,170,BUF,Jordan Hancock,SAF,22,2025,0,0,0,1,1,13,0,0,0,0,0,0,0,0,0,0,0,9,,,Ohio St.,College Stats,HancJo00
5,171,DET,Miles Frazier,G,23,2025,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,,,,LSU,College Stats,FrazMi00
5,172,LAR,Chris Paul,LB,22,,0,0,0,,,,,,,,,,,,,,,,,,Mississippi,College Stats,PaulCh01
5,173,BUF,Jackson Hawes,TE,24,2025,0,0,0,2,2,17,0,0,0,0,0,0,0,0,16,187,3,,,,Georgia Tech,College Stats,HaweJa00
5,174,ARI,Denzel Burke,CB,22,2025,0,0,0,3,3,17,0,0,0,0,0,0,0,0,0,0,0,34,3,,Ohio St.,College Stats,BurkDe01
5,175,SEA,Robbie Ouzts,TE,23,2025,0,0,0,0,0,12,0,0,0,0,0,0,0,0,0,0,0,,,,Alabama,College Stats,OuztRo00
5,176,NYJ,Tyler Baron,DE,23,2025,0,0,0,0,0,6,0,0,0,0,0,0,0,0,0,0,0,5,,,Miami (FL),College Stats,BaroTy00
6,177,BUF,Dorian Strong,CB,23,2025,0,0,0,1,1,4,0,0,0,0,0,0,0,0,0,0,0,9,,,Virginia Tech,College Stats,StroDo01
6,178,BAL,Bilhal Kone,CB,23,,0,0,0,,,,,,,,,,,,,,,,,,Western Michigan,College Stats,KoneBi00
6,179,MIA,Ollie Gordon,RB,21,2025,0,0,0,2,2,17,0,0,0,0,0,70,199,3,7,32,1,2,,,Oklahoma St.,College Stats,GordOl00
6,180,LVR,JJ Pegues,DT,23,2025,0,0,0,1,1,9,0,0,0,0,0,0,0,0,0,0,0,4,,,Mississippi,College Stats,PeguJJ00
6,181,PHI,Kyle McCord,QB,22,,0,0,0,,,,,,,,,,,,,,,,,,Syracuse,College Stats,McCoKy00
6,182,NWE,Andres Borregales,K,22,2025,0,0,0,2,2,17,0,0,0,0,0,0,0,0,0,0,0,1,,,Miami (FL),College Stats,BorrAn00
6,183,TEN,Marcus Harris,DB,24,2025,0,0,0,2,2,14,0,0,0,0,0,0,0,0,0,0,0,18,,,California,College Stats,HarrMa10
6,184,NOR,Devin Neal,RB,22,2025,0,0,0,2,2,10,0,0,0,0,0,57,206,2,17,104,0,,,,Kansas,College Stats,NealDe00
6,185,PIT,Will Howard,QB,23,,0,0,0,,,,,,,,,,,,,,,,,,Ohio St.,College Stats,HowaWi01
6,186,BAL,Tyler Loop,K,24,2025,0,0,0,3,3,17,0,0,0,0,0,0,0,0,0,0,0,2,,,Arizona,College Stats,LoopTy00
6,187,HOU,Jaylen Reed,SAF,22,2025,0,0,0,1,1,7,0,0,0,0,0,0,0,0,0,0,0,9,,,Penn St.,College Stats,ReedJa04
6,188,TEN,Kalel Mullings,RB,22,2025,0,0,0,0,0,9,0,0,0,0,0,3,7,0,0,0,0,1,,,Michigan,College Stats,MullKa00
6,189,IND,Riley Leonard,QB,22,2025,0,0,0,2,2,5,39,67,415,2,3,6,27,2,0,-3,0,,,,Notre Dame,College Stats,LeonRi02
6,190,IND,Tim Smith,DL,23,,0,0,0,,,,,,,,,,,,,,,,,,Alabama,College Stats,SmitTi02
6,191,PHI,Myles Hinton,OT,23,,0,0,0,,,,,,,,,,,,,,,,,,Michigan,College Stats,HintMy00
6,192,SEA,Bryce Cabeldue,OT,24,2025,0,0,0,1,1,8,0,0,0,0,0,0,0,0,0,0,0,,,,Kansas,College Stats,CabeBr00
6,193,CIN,Tahj Brooks,RB,23,2025,0,0,0,0,0,16,0,0,0,0,0,16,45,0,1,9,0,5,,,Texas Tech,College Stats,BrooTa00
6,194,JAX,Jalen McLeod,DE,23,,0,0,0,,,,,,,,,,,,,,,,,,Auburn,College Stats,McLeJa00
6,195,CHI,Luke Newman,G,23,2025,0,0,0,1,1,9,0,0,0,0,0,0,0,0,0,0,0,,,,Michigan St.,College Stats,NewmLu00
6,196,DET,Ahmed Hassanein,DE,23,,0,0,0,,,,,,,,,,,,,,,,,,Boise St.,College Stats,HassAh00
6,197,HOU,Graham Mertz,QB,24,,0,0,0,,,,,,,,,,,,,,,,,,Florida,College Stats,MertGr00
6,198,GNB,Warren Brinson,DL,23,2025,0,0,0,1,1,11,0,0,0,0,0,0,0,0,0,0,0,6,,0.5,Georgia,College Stats,BrinWa00
6,199,LAC,Branson Taylor,OL,23,2025,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,,,,Pittsburgh,College Stats,TaylBr04
6,200,JAX,Rayuan Lane,FS,22,2025,0,0,0,1,1,17,0,0,0,0,0,0,0,0,0,0,0,12,,,Navy,College Stats,LaneRa00
6,201,MIN,Kobe King,LB,22,2025,0,0,0,1,0,14,0,0,0,0,0,0,0,0,0,0,0,5,,,Penn St.,College Stats,KingKo00
6,202,MIN,Gavin Bartholomew,TE,22,,0,0,0,,,,,,,,,,,,,,,,,,Pittsburgh,College Stats,BartGa00
6,203,BAL,LaJohntay Wester,WR,23,2025,0,0,0,0,0,17,0,0,0,0,0,0,0,0,0,0,0,,,,Colorado,College Stats,WestLa01
6,204,DAL,Ajani Cornelius,OL,23,2025,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,,,,Oregon,College Stats,CornAj00
6,205,WAS,Kain Medrano,LB,24,2025,0,0,0,1,1,9,0,0,0,0,0,0,0,0,0,0,0,3,,,UCLA,College Stats,MedrKa00
6,206,BUF,Chase Lundt,OL,25,2025,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,,,,Connecticut,College Stats,LundCh00
6,207,PHI,Cameron Williams,OL,21,2025,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,,,,Texas,College Stats,WillCa04
6,208,CAR,Jimmy Horn,WR,22,2025,0,0,0,1,1,13,0,0,0,0,0,8,69,0,11,108,0,,,,Colorado,College Stats,HornJi00
6,209,PHI,Antwaun Powell-Ryland,DE,23,,0,0,0,,,,,,,,,,,,,,,,,,Virginia Tech,College Stats,PoweAn00
6,210,BAL,Aeneas Peebles,DT,24,2025,0,0,0,0,0,6,0,0,0,0,0,0,0,0,0,0,0,2,,,Virginia Tech,College Stats,PeebAe00
6,211,ARI,Hayden Conner,OL,23,2025,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,,,,Texas,College Stats,ConnHa00
6,212,BAL,Robert Longerbeam,DB,24,,0,0,0,,,,,,,,,,,,,,,,,,Rutgers,College Stats,LongRo00
6,213,LVR,Tommy Mellott,QB,,,0,0,0,,,,,,,,,,,,,,,,,,Montana St.,,MellTo00
6,214,LAC,R.J. Mickens,SAF,24,2025,0,0,0,3,3,12,0,0,0,0,0,0,0,0,0,0,0,18,2,,Clemson,College Stats,MickRJ00
6,215,LVR,Cam Miller,QB,24,,0,0,0,,,,,,,,,,,,,,,,,,North Dakota St.,,MillCa00
6,216,DEN,Jeremy Crawshaw,P,24,2025,0,0,0,3,3,17,0,0,0,0,0,0,0,0,0,0,0,,,,Florida,College Stats,CrawJe00
7,217,DAL,Jay Toia,DT,22,2025,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,1,,,UCLA,College Stats,ToiaJa00
7,218,ATL,Jack Nelson,OT,23,2025,0,0,0,1,1,10,0,0,0,0,0,0,0,0,0,0,0,,,,Wisconsin,College Stats,NelsJa01
7,219,NYG,Thomas Fidone,TE,22,2025,0,0,0,0,0,7,0,0,0,0,0,0,0,0,0,0,0,1,,,Nebraska,College Stats,FidoTh00
7,220,NWE,Marcus Bryant,OL,23,2025,0,0,0,1,1,12,0,0,0,0,0,0,0,0,0,0,0,,,,Missouri,College Stats,BryaMa02
7,221,JAX,Jonah Monheim,OL,23,2025,0,0,0,2,2,17,0,0,0,0,0,0,0,0,0,0,0,,,,USC,College Stats,MonhJo00
7,222,LVR,Cody Lindenberg,LB,23,2025,0,0,0,1,1,17,0,0,0,0,0,0,0,0,0,0,0,4,,,Minnesota,College Stats,LindCo00
7,223,SEA,Damien Martinez,RB,21,,0,0,0,,,,,,,,,,,,,,,,,,Miami (FL),College Stats,MartDa02
7,224,HOU,Kyonte Hamilton,DL,22,,0,0,0,,,,,,,,,,,,,,,,,,Rutgers,College Stats,HamiKy01
7,225,ARI,Kitan Crawford,SAF,23,2025,0,0,0,1,1,15,0,0,0,0,0,0,0,0,0,0,0,11,,,Nevada,College Stats,CrawKi00
7,226,PIT,Carson Bruener,LB,24,2025,0,0,0,1,1,17,0,0,0,0,0,0,0,0,0,0,0,7,,,Washington,College Stats,BrueCa00
7,227,SFO,Kurtis Rourke,QB,24,,0,0,0,,,,,,,,,,,,,,,,,,Indiana,College Stats,RourKu01
7,228,KAN,Brashard Smith,RB,22,2025,0,0,0,3,3,17,0,0,0,0,0,44,151,0,25,172,1,,,,SMU,College Stats,SmitBr09
7,229,PIT,Donte Kent,CB,23,,0,0,0,,,,,,,,,,,,,,,,,,Central Michigan,College Stats,KentDo00
7,230,DET,Dan Jackson,SAF,24,,0,0,0,,,,,,,,,,,,,,,,,,Georgia,College Stats,JackDa04
7,231,MIA,Quinn Ewers,QB,22,2025,0,0,0,2,2,4,55,83,622,3,3,8,19,0,1,-6,0,,,,Texas,College Stats,EwerQu00
7,232,IND,Hunter Wohler,SAF,22,,0,0,0,,,,,,,,,,,,,,,,,,Wisconsin,College Stats,WohlHu00
7,233,CHI,Kyle Monangai,RB,23,2025,0,0,0,6,6,17,0,0,0,0,0,169,783,5,18,164,0,,,,Rutgers,College Stats,MonaKy00
7,234,SEA,Mason Richman,OL,23,2025,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,,,,Iowa,College Stats,RichMa01
7,235,TAM,Tez Johnson,WR,23,2025,0,0,0,3,3,16,0,0,0,0,0,7,22,0,28,322,5,,,,Oregon,College Stats,JohnTe03
7,236,JAX,LeQuint Allen,RB,21,2025,0,0,0,1,1,17,0,0,0,0,0,23,94,0,10,54,0,1,,,Syracuse,College Stats,AlleLe01
7,237,GNB,Micah Robinson,CB,23,2025,0,0,0,1,,9,0,0,0,0,0,0,0,0,0,0,0,8,,,Tulane,College Stats,RobiMi01
7,238,SEA,Ricky White,WR,23,2025,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,,,,UNLV,College Stats,WhitRi01
7,239,DAL,Phil Mafah,RB,22,2025,0,0,0,0,0,1,0,0,0,0,0,5,18,1,2,11,0,,,,Clemson,College Stats,MafaPh00
7,240,BUF,Kaden Prather,WR,23,,0,0,0,,,,,,,,,,,,,,,,,,Maryland,College Stats,PratKa00
7,241,DEN,Caleb Lohner,TE,24,,0,0,0,,,,,,,,,,,,,,,,,,Utah,College Stats,LohnCa00
7,242,LAR,Konata Mumpfield,WR,22,2025,0,0,0,1,1,17,0,0,0,0,0,0,0,0,10,92,1,,,,Pittsburgh,College Stats,MumpKo00
7,243,BAL,Garrett Dellinger,G,23,2025,0,0,0,0,,1,0,0,0,0,0,0,0,0,0,0,0,,,,LSU,College Stats,DellGa00
7,244,DET,Dominic Lovett,WR,22,2025,0,0,0,0,0,12,0,0,0,0,0,0,0,0,0,0,0,,,,Georgia,College Stats,LoveDo00
7,245,WAS,Jacory Croskey-Merritt,RB,24,2025,0,0,0,6,6,17,0,0,0,0,0,175,805,8,9,68,0,,,,Arizona,College Stats,CrosJa00
7,246,NYG,Korie Black,CB,23,2025,0,0,0,1,1,11,0,0,0,0,0,0,0,0,0,0,0,7,,,Oklahoma St.,College Stats,BlacKo00
7,247,DAL,Tommy Akingbesote,DT,22,,0,0,0,,,,,,,,,,,,,,,,,,Maryland,College Stats,AkinTo00
7,248,NOR,Moliki Matavao,TE,22,2025,0,0,0,0,0,6,0,0,0,0,0,0,0,0,2,10,0,,,,UCLA,College Stats,MataMo00
7,249,SFO,Connor Colby,OL,22,2025,0,0,0,3,3,13,0,0,0,0,0,0,0,0,1,-1,0,,,,Iowa,College Stats,ColbCo00
7,250,GNB,John Williams,OL,23,,0,0,0,,,,,,,,,,,,,,,,,,Cincinnati,College Stats,WillJo11
7,251,NWE,Julian Ashby,LS,23,2025,0,0,0,1,1,17,0,0,0,0,0,0,0,0,0,0,0,2,,,Vanderbilt,College Stats,AshbJu00
7,252,SFO,Junior Bergen,WR,,,0,0,0,,,,,,,,,,,,,,,,,,Montana,,BergJu00
7,253,MIA,Zeek Biggers,DL,21,2025,0,0,0,1,1,9,0,0,0,0,0,0,0,0,0,0,0,6,,1.5,Georgia Tech,College Stats,BiggZe01
7,254,NOR,Fadil Diggs,DE,23,2025,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,,,,Syracuse,College Stats,DiggFa00
7,255,HOU,Luke Lachey,TE,24,,0,0,0,,,,,,,,,,,,,,,,,,Iowa,College Stats,LachLu00
7,256,LAC,Trikweze Bridges,SAF,24,2025,0,0,0,1,,16,0,0,0,0,0,0,0,0,0,0,0,17,1,,Florida,College Stats,BridTr00
7,257,NWE,Kobee Minor,DB,23,2025,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,1,,,Memphis,College Stats,MinoKo00"""

# Parse and write
def parse_row(line: str) -> list:
    parts = line.strip().split(",")
    if len(parts) < 6:
        return []
    rnd, pick, tm, player, pos = parts[0], parts[1], parts[2], parts[3], parts[4]
    school = parts[-3] if len(parts) >= 3 else ""
    return [rnd, pick, tm, player, pos, school]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "2025_draft_picks.csv")
    rows = [["Rnd", "Pick", "Tm", "Player", "Pos", "School"]]
    for line in RAW_TABLE.strip().split("\n"):
        row = parse_row(line)
        if row:
            rows.append(row)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"Wrote {len(rows)-1} picks to {out_path}")

if __name__ == "__main__":
    main()
