"""Two real score targets, not a universal conversion between judgement mixes.

PP uses 320-weighted accuracy. The screen uses 305-weighted accuracy. Their
conditional expectations differ with OD, holds and the player's judgement mix.
Both are fitted from the same saved counts; neither changes the recorded score.
"""
import math

FIELDS=('mu','base_mu','cold_penalty','sdm','sda','sd')


def log_loss(accuracy):
    return math.log(max(1e-3,1.-min(float(accuracy),.999)))


def session_view(session, warmup_model):
    """Shared completed effort, but each target consumes its own frozen residuals.

    Legacy predictions still establish warmth. They are not silently interpreted
    as directly fitted display forecasts; their saved scores remain available to
    the new target's historical fit.
    """
    import recommend
    cache=session.__dict__.setdefault('_accuracy_views',{})
    key=(id(warmup_model),session.warm_flag)
    if key in cache:return cache[key]
    events=[]
    for event in session.all:
        info=dict(event['info'])
        if event['kind']=='start':
            for field in FIELDS:info[field]=info.get('display_'+field)
            info['shown_mu']=info['mu']
        elif event['kind']=='finish':
            info['y']=info.get('y_lazer')
        events.append(dict(event,info=info))
    view=recommend.Session(events,session.now,warmup_model=warmup_model)
    view.warm_flag=session.warm_flag
    cache[key]=view
    return view
