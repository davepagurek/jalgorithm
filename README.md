# A Jalgorithm for Japplying Jeans to Jobjects

![teaser](https://user-images.githubusercontent.com/5315059/224508329-3d7a5370-3eb7-4d84-b76f-9811aec71f2a.png)

<a href="https://davepagurek.github.io/jalgorithm/paper/jalgorithm.pdf">Read the paper</a>

For years, society has been plagued by the lack of a conclusive answer to the question of how arbitrary things would wear jeans, if they wore jeans. This paper proposes a jalgorithm to determine the mathematically optimal configuration of pants on a silhouette, leveraging recent insights in the fields of computer vision and differentiable rendering.

This is based off of the <a href="https://github.com/BachiLi/diffvg">diffvg differential vector graphics renderer</a>. It uses a customized loss function to optimize the vertices of pants, based on the following rules:

1. **Duality of man:** Pants should bisect the silhouette of the wearer through its center, covering half its area.
2. **No-stretch jeans:** Pants, when applied to the wearer in rest position, should not be so deformed that they no longer resemble prototypical jeans.
3. **(Topologically) torn jean avoidance:** Limbs should only go through the open ends of pants, and similarly, closed ends of pants should not intersect limbs.
4. **Dress code compliance:** Pants should be on the body of the wearer, not off of it.

<a href="https://davepagurek.github.io/jalgorithm/paper/jalgorithm.pdf">Read the paper for more details!</a>

You really shouldn't look at the code for this as I wrote it at 1am really quickly and hackily, but if you really want to, it's in `apps/pants.py`.

## Results

<table>
<tr>
<td>
<img src="https://user-images.githubusercontent.com/5315059/224508378-37ec4611-b07d-48ec-91db-497c5fc87a8c.gif" />
</td>
<td>
<img src="https://user-images.githubusercontent.com/5315059/224508390-97085d19-40a5-4fde-a6a0-3937756bf6fd.gif" />
</td>
<td>
<img src="https://user-images.githubusercontent.com/5315059/224508416-462cce71-6744-4943-a893-425b6740978b.gif" />
</td>
<td>
<img src="https://user-images.githubusercontent.com/5315059/224508425-cafd58aa-539a-48e4-b5de-242fbf92ad2e.gif" />
</td>
</tr>
</table>


## Running the Code

1. Follow the setup instructions from <a href="https://github.com/BachiLi/diffvg">BachiLi/diffvg</a>
2. `cd apps`
3. `python pants.py`
