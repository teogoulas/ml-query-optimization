SELECT k.phonetic_code AS movie_phonetic_code,
       t.title AS marvel_movie
FROM keyword AS k,
     movie_keyword AS mk,
     title AS t
WHERE k.id = mk.keyword_id
  AND t.id = mk.movie_id
AND k.keyword like '%ironman%';