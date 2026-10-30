"""Two real score targets, not a universal conversion between judgement mixes.

PP uses 320-weighted accuracy. The screen uses 305-weighted accuracy. Their
conditional expectations differ with OD, holds and the player's judgement mix.
Both are fitted from the same saved counts; neither changes the recorded score.
"""
import math

FIELDS=('mu','base_mu','cold_penalty','cold_scale','sdm','sda','sd')


def log_loss(accuracy):
    return math.log(max(1e-3,1.-min(float(accuracy),.999)))


def forecast_state(prediction, *, generated_at, calculator, predictor, rate, band_z,
                   sha256=None, md5=None):
    """Copy the directly fitted display state; a PP fallback has no native state."""
    if any(prediction.get('display_'+k) is None for k in ('mu','sdm','sda','sd')):
        return None
    return dict(schema=1, target='native_displayed_lazer', kind='usual_range',
                generated_at=generated_at, calculator=calculator, predictor=predictor,
                sha256=sha256, md5=md5, rate=rate, band_z=band_z,
                **{k:prediction['acc_'+k] for k in ('mid','lo','hi')},
                **{k:prediction['display_'+k] for k in ('mu','sdm','sda','sd')})


def forecast_chart(state, sha256, md5):
    """Attach an exact chart identity without changing prediction provenance."""
    if state is None:
        return None
    identities = dict(sha256=sha256, md5=md5)
    if any(state.get(k) is not None and v is not None and state[k]!=v for k,v in identities.items()):
        return None
    return dict(state, **{k:state[k] if state.get(k) is not None else v for k,v in identities.items()})


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
            legacy_scale=info.get('cold_scale')
            for field in FIELDS:info[field]=info.get('display_'+field)
            if info['cold_scale'] is None:
                # The old paired writer saved the display scale in the common
                # field. Its PP scale is unknown, but this target can retain it.
                info['cold_scale']=legacy_scale
            info['shown_mu']=info['mu']
        elif event['kind']=='finish':
            info['y']=info.get('y_lazer')
        events.append(dict(event,info=info))
    view=recommend.Session(events,session.now,warmup_model=warmup_model)
    view._display_target=True
    view.warm_flag=session.warm_flag
    cache[key]=view
    return view
